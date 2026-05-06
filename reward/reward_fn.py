"""
reward/reward_fn.py
===================
Combined reward function for the SWE-RL re-implementation.

Original SWE-RL reward  (Equation 1 in the paper):
    R(o) = -1                                 if wrong format
    R(o) = compare(patch_pred, patch_gt)      otherwise  in (0, 1)

Re-implementation reward  (midway report §3.3):
    R(o) = -1                                 if wrong format
    R(o) = alpha * correctness(o) + (1-alpha) * sim(o)   otherwise

    correctness(o) in {0, 1}:
        1 iff (a) patch applies cleanly to file_contents AND
              (b) every modified file has valid Python syntax AND
              (c) at least one file was actually changed AND
              (d) no new flake8 fatal errors were introduced

    sim(o) in [0, 1]:
        Sequence similarity between the normalised predicted patch
        and the oracle patch  (= upstream's calculate_search_replace_reward)

    Default alpha = 0.3  (set in configs/train_config.yaml).
"""

import ast
import json
import logging
import os
import sys
from pathlib import Path

from utils.git_utils import check_code_differ_by_just_empty_lines, lint_code

logger = logging.getLogger(__name__)

# Add swe-rl-main to path so we can import the upstream reward utilities
_SWERL_SRC = Path(__file__).parents[2] / "swe-rl-main" / "src"
if _SWERL_SRC.exists() and str(_SWERL_SRC) not in sys.path:
    sys.path.insert(0, str(_SWERL_SRC))

try:
    from swerl.core.reward import (
        calculate_search_replace_reward,
        apply_code_change,
        parse_search_replace,
        extract_thought_solution,
        FormatError,
    )
    _UPSTREAM_AVAILABLE = True
except ImportError:
    logger.warning(
        "swerl.core.reward not found on sys.path — using bundled fallback."
    )
    _UPSTREAM_AVAILABLE = False

    import difflib
    import re

    THINK_START, THINK_END = "<think>", "</think>"
    ANSWER_START, ANSWER_END = "<solution>", "</solution>"
    SR_REGEX = (
        r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"
    )

    class FormatError(Exception):
        pass

    def extract_thought_solution(output):
        for tag in [THINK_START, THINK_END, ANSWER_START, ANSWER_END]:
            if output.count(tag) != 1:
                raise FormatError(f"count of {tag} is not 1")
        thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
        answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
        if not thought:
            raise FormatError("Thought is empty")
        return thought, answer

    def parse_search_replace(text):
        results = {}
        for path, search, replace in re.findall(SR_REGEX, text):
            results.setdefault(path, []).append((search, replace))
        return results

    def apply_code_change(code_context, sr_dict, silent=False):
        new_content = {}
        for path, pairs in sr_dict.items():
            content = "\n" + code_context.get(path, "")
            for search, replace in pairs:
                if not silent and search == replace:
                    raise FormatError("Search and replace blocks are identical")
                s, r = "\n" + search, "\n" + replace
                if not silent and s not in content:
                    raise FormatError(f"Search block not found: {s}")
                content = content.replace(s, r)
            new_content[path] = content[1:]
        return new_content

    def _get_patch(code_context, new_content_dict):
        patches = {}
        for path, new in new_content_dict.items():
            old = code_context.get(path, "")
            diff = list(difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm="", n=3
            ))
            if diff:
                patches[path] = "\n".join(diff[2:])
        return patches

    def calculate_search_replace_reward(code_context, oracle_new_content, output):
        try:
            thought, answer = extract_thought_solution(output)
            sr = parse_search_replace(answer)
            if not sr:
                raise FormatError("No SEARCH/REPLACE blocks found")
            pred_new = apply_code_change(code_context, sr)
            oracle_p = _get_patch(code_context, oracle_new_content)
            pred_p = _get_patch(code_context, pred_new)
            all_paths = set(oracle_p) | set(pred_p)
            if not all_paths:
                return 1.0, {"similarities": []}
            sims = []
            for p in all_paths:
                pc, oc = pred_p.get(p, ""), oracle_p.get(p, "")
                sims.append(
                    0.0 if not pc or not oc
                    else difflib.SequenceMatcher(None, pc, oc, autojunk=False).ratio()
                )
            reward = sum(sims) / len(sims)
            return reward, {"similarities": sims, "thought": thought, "answer": answer}
        except FormatError as e:
            return -1.0, {"error": str(e)}

DEFAULT_ALPHA = 0.3
FORMAT_PENALTY = -1.0
PLAYGROUND_DIR = os.getenv("PLAYGROUND_DIR", "playground")


def check_correctness(
    code_context: dict,
    pred_new_content: dict,
    use_git_apply: bool = True,
    use_lint: bool = True,
) -> tuple:
    """
    Check whether the predicted file changes are correct in a syntactic sense.

    Returns (score in {0.0, 1.0}, metadata).

    Correctness conditions (all must hold):
        1. At least one file was actually changed
        2. Every modified file produces valid Python syntax
        3. The changes are not merely empty-line additions
        4. No new flake8 fatal errors introduced  (if use_lint=True)
    """
    meta = {"correctness_checks": {}}

    if not pred_new_content:
        meta["correctness_error"] = "No files modified"
        return 0.0, meta

    modified_files = {}
    for path, new_content in pred_new_content.items():
        original = code_context.get(path, "")
        if new_content != original:
            modified_files[path] = (original, new_content)

    # Check 1: At least one real change
    if not modified_files:
        meta["correctness_error"] = "Predicted content identical to original (no-op)"
        meta["correctness_checks"]["changed"] = False
        return 0.0, meta
    meta["correctness_checks"]["changed"] = True

    # Check 2: Not just empty-line changes
    originals = [o for o, _ in modified_files.values()]
    new_codes = [n for _, n in modified_files.values()]
    if check_code_differ_by_just_empty_lines(new_codes, originals):
        meta["correctness_error"] = "Only empty-line differences (trivial patch)"
        meta["correctness_checks"]["non_trivial"] = False
        return 0.0, meta
    meta["correctness_checks"]["non_trivial"] = True

    # Check 3: Valid Python syntax for modified Python files.
    syntax_errors = []
    for path, (_, new_content) in modified_files.items():
        if not str(path).endswith(".py"):
            continue
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            syntax_errors.append(f"{path}: line {e.lineno}: {e.msg}")
    if syntax_errors:
        meta["correctness_error"] = "Syntax errors: " + "; ".join(syntax_errors)
        meta["correctness_checks"]["syntax_ok"] = False
        return 0.0, meta
    meta["correctness_checks"]["syntax_ok"] = True

    # Check 4: No new flake8 fatal errors (optional, requires flake8)
    if use_lint:
        lint_errors = []
        for path, (old_content, new_content) in modified_files.items():
            filename = Path(path).name
            try:
                os.makedirs(PLAYGROUND_DIR, exist_ok=True)
                passed, prev_errs, new_errs = lint_code(
                    PLAYGROUND_DIR, filename, new_content, old_content
                )
                if not passed:
                    added = new_errs - prev_errs
                    lint_errors.append(f"{path}: {added}")
            except Exception as e:
                # flake8 might not be installed; skip gracefully
                logger.debug(f"lint_code failed for {path}: {e}")
        if lint_errors:
            meta["correctness_error"] = "New lint errors: " + "; ".join(lint_errors)
            meta["correctness_checks"]["lint_ok"] = False
            return 0.0, meta
    meta["correctness_checks"]["lint_ok"] = True

    return 1.0, meta


def calculate_combined_reward(
    code_context: dict,
    oracle_new_content: dict,
    output: str,
    alpha: float = DEFAULT_ALPHA,
    use_lint: bool = True,
) -> tuple:
    """
    Full combined reward for a single rollout output.

    R(o) = -1                               if format error
    R(o) = alpha * correctness + (1-alpha) * sim    otherwise

    Args:
        code_context       : {path: original_content} before patch
        oracle_new_content : {path: patched_content}  ground truth
        output             : Raw LLM string (<think>...</think><solution>...</solution>)
        alpha              : Correctness weight (0 = pure similarity)
        use_lint           : Whether to run flake8 in correctness check

    Returns:
        (reward, metadata)  where reward in {-1} union [0, 1]
    """
    # Step 1: Upstream reward (format + similarity)
    sim_score, sim_meta = calculate_search_replace_reward(
        code_context=code_context,
        oracle_new_content=oracle_new_content,
        output=output,
    )

    # Step 2: Format error?
    if sim_score < 0:
        return FORMAT_PENALTY, sim_meta

    # Step 3: Re-extract pred_new_content for correctness check
    try:
        _, answer = extract_thought_solution(output)
        sr_dict = parse_search_replace(answer)
        pred_new_content = apply_code_change(code_context, sr_dict, silent=True)
    except FormatError:
        return FORMAT_PENALTY, {"error": "re-parse failed after sim_score >= 0"}

    # Step 4: Correctness
    correctness_score, correctness_meta = check_correctness(
        code_context=code_context,
        pred_new_content=pred_new_content,
        use_lint=use_lint,
    )

    # Step 5: Combine
    reward = alpha * correctness_score + (1.0 - alpha) * sim_score

    meta = {
        **sim_meta,
        **correctness_meta,
        "sim_score": sim_score,
        "correctness_score": correctness_score,
        "alpha": alpha,
        "combined_reward": reward,
    }
    return reward, meta


def calculate_rewards_batch(
    code_context: dict,
    oracle_new_content: dict,
    outputs: list,
    alpha: float = DEFAULT_ALPHA,
) -> tuple:
    """Compute combined rewards for a batch of G rollouts (one problem)."""
    rewards, metas = [], []
    for output in outputs:
        r, m = calculate_combined_reward(code_context, oracle_new_content, output, alpha)
        rewards.append(r)
        metas.append(m)
    return rewards, metas


class SWERLRewardFunction:
    """
    Callable reward adapter for trl's GRPOTrainer.

    GRPOTrainer calls:
        reward_fn(completions, **extra_columns_from_dataset)

    The dataset must include columns:
        code_context        (JSON-encoded {path: original_content})
        oracle_new_content  (JSON-encoded {path: patched_content})
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, use_lint: bool = True):
        self.alpha = alpha
        self.use_lint = use_lint

    @staticmethod
    def _coerce_list(value, n: int) -> list:
        """Normalize TRL scalar/list reward kwargs to one item per completion."""
        if value is None:
            return [None] * n
        if not isinstance(value, list):
            return [value] * n
        if len(value) == n:
            return value
        if len(value) == 1:
            return value * n

        # Some trainer versions pass one metadata row per prompt and a flattened
        # completion list. Repeat rows evenly when possible.
        if n % len(value) == 0:
            repeat = n // len(value)
            expanded = []
            for item in value:
                expanded.extend([item] * repeat)
            return expanded

        logger.warning(
            "Reward metadata length mismatch: %s metadata rows for %s completions; "
            "falling back to index clamping.",
            len(value),
            n,
        )
        return [value[min(i, len(value) - 1)] for i in range(n)]

    @staticmethod
    def _loads_if_json(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    @staticmethod
    def _completion_to_text(completion) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            parts = []
            for item in completion:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if isinstance(completion, dict):
            return str(completion.get("content", completion))
        return str(completion)

    def __call__(
        self,
        completions: list,
        code_context=None,
        oracle_new_content=None,
        **kwargs,
    ) -> list:
        if code_context is None or oracle_new_content is None:
            logger.warning("Reward fn missing code_context / oracle_new_content")
            return [FORMAT_PENALTY] * len(completions)

        contexts = self._coerce_list(code_context, len(completions))
        oracles = self._coerce_list(oracle_new_content, len(completions))

        rewards = []
        for idx, (completion, ctx_raw, oracle_raw) in enumerate(
            zip(completions, contexts, oracles)
        ):
            try:
                ctx = self._loads_if_json(ctx_raw)
                oracle = self._loads_if_json(oracle_raw)
                reward, _ = calculate_combined_reward(
                    code_context=ctx,
                    oracle_new_content=oracle,
                    output=self._completion_to_text(completion),
                    alpha=self.alpha,
                    use_lint=self.use_lint,
                )
            except Exception as e:
                logger.warning("Reward failed for completion %s: %s", idx, e)
                reward = FORMAT_PENALTY
            rewards.append(reward)

        return rewards
