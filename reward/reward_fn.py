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

# Import difflib for continuous correctness scoring
import difflib


def compute_patch_similarity_correctness(
    code_context: dict,
    pred_new_content: dict,
    oracle_new_content: dict,
) -> tuple:
    """
    Compute continuous correctness score using SequenceMatcher.

    Measures how similar the predicted changes are to the oracle (ground truth).
    This gives partial credit for patches that are "close" to correct.

    Returns:
        (correctness_score in [0, 1], metadata)

    Scoring:
        - 0.0: No changes or completely wrong
        - 0.3-0.7: Partial changes in right direction
        - 1.0: Matches oracle exactly
    """
    meta = {}

    if not pred_new_content:
        return 0.0, {"error": "No predicted content"}

    similarities = []

    # Compare each predicted file against oracle
    for path, pred_content in pred_new_content.items():
        oracle_content = oracle_new_content.get(path, "")

        # Use SequenceMatcher to compute similarity
        matcher = difflib.SequenceMatcher(None, oracle_content, pred_content)
        similarity = matcher.ratio()  # Returns [0, 1]
        similarities.append(similarity)

        meta[f"{path}_similarity"] = similarity

    if not similarities:
        return 0.0, {"error": "No files to compare"}

    # Average similarity across all modified files
    avg_similarity = sum(similarities) / len(similarities)

    meta["patch_similarity_correctness"] = avg_similarity
    meta["num_files_compared"] = len(similarities)

    return avg_similarity, meta


def check_correctness(
    code_context: dict,
    pred_new_content: dict,
    use_git_apply: bool = True,
    use_lint: bool = True,
    continuous: bool = True,
) -> tuple:
    """
    Check whether the predicted file changes are correct.

    Returns (score, metadata).

    If continuous=True (default):
        Returns continuous score in [0, 1] with partial credit.
    If continuous=False:
        Returns binary score in {0.0, 1.0} (original behavior).

    Continuous scoring (recommended per SWE-RL paper):
        0.00 - No files modified or only trivial changes
        0.25 - Files changed but syntax errors present
        0.50 - Syntax OK but lint errors present
        0.75 - Syntax + lint OK but other minor issues
        1.00 - Fully correct (no syntax/lint/format errors)
    """
    meta = {"correctness_checks": {}}

    if not pred_new_content:
        meta["correctness_error"] = "No files modified"
        return (0.0, meta) if not continuous else (0.0, meta)

    modified_files = {}
    for path, new_content in pred_new_content.items():
        original = code_context.get(path, "")
        if new_content != original:
            modified_files[path] = (original, new_content)

    # Check 1: At least one real change
    if not modified_files:
        meta["correctness_error"] = "Predicted content identical to original (no-op)"
        meta["correctness_checks"]["changed"] = False
        return (0.0, meta) if not continuous else (0.0, meta)
    meta["correctness_checks"]["changed"] = True

    # Check 2: Not just empty-line changes
    originals = [o for o, _ in modified_files.values()]
    new_codes = [n for _, n in modified_files.values()]
    if check_code_differ_by_just_empty_lines(new_codes, originals):
        meta["correctness_error"] = "Only empty-line differences (trivial patch)"
        meta["correctness_checks"]["non_trivial"] = False
        return (0.0, meta) if not continuous else (0.0, meta)
    meta["correctness_checks"]["non_trivial"] = True

    # Starting score for continuous mode
    score = 0.25 if continuous else 0.0

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
        if not continuous:
            return 0.0, meta
        # Continuous: partial credit for valid attempt
        score = 0.25
        meta["correctness_score_intermediate"] = score
    else:
        meta["correctness_checks"]["syntax_ok"] = True
        score = 0.5 if continuous else 1.0

    # Check 4: No new flake8 fatal errors (optional, requires flake8)
    if use_lint and not syntax_errors:  # Only check lint if syntax is OK
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
            if not continuous:
                return 0.0, meta
            # Continuous: syntax OK, but lint issues = 0.5-0.75
            score = 0.75 if continuous else 0.0
            meta["correctness_score_intermediate"] = score
        else:
            meta["correctness_checks"]["lint_ok"] = True
            score = 1.0

    return score, meta


def calculate_combined_reward(
    code_context: dict,
    oracle_new_content: dict,
    output: str,
    alpha: float = DEFAULT_ALPHA,
    use_lint: bool = True,
    continuous_correctness: bool = True,
    use_matcher_correctness: bool = True,
) -> tuple:
    """
    Full combined reward for a single rollout output.

    R(o) = -1                               if format error
    R(o) = alpha * correctness + (1-alpha) * sim    otherwise

    Correctness scoring modes (recommended: use_matcher_correctness=True):
    - use_matcher_correctness=True (default):
        Continuous [0, 1] using SequenceMatcher similarity to oracle.
        Rewards patches that are "close" to correct.
    - use_matcher_correctness=False:
        Binary {0, 1} based on syntax/lint checks.
        All-or-nothing scoring.

    Args:
        code_context          : {path: original_content} before patch
        oracle_new_content    : {path: patched_content}  ground truth
        output                : Raw LLM string (<think>...</think><solution>...</solution>)
        alpha                 : Correctness weight (0 = pure similarity, 1 = pure correctness)
        use_lint              : Whether to run flake8 in correctness check (ignored if use_matcher_correctness=True)
        continuous_correctness: Legacy parameter (ignored if use_matcher_correctness=True)
        use_matcher_correctness: Use SequenceMatcher for continuous correctness (recommended)

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

    # Step 4: Correctness scoring
    if use_matcher_correctness:
        # NEW: Use SequenceMatcher for continuous correctness
        # This measures how similar predicted changes are to oracle
        correctness_score, correctness_meta = compute_patch_similarity_correctness(
            code_context=code_context,
            pred_new_content=pred_new_content,
            oracle_new_content=oracle_new_content,
        )
        correctness_mode = "patch_similarity"
    else:
        # OLD: Use syntax/lint checks for binary correctness
        correctness_score, correctness_meta = check_correctness(
            code_context=code_context,
            pred_new_content=pred_new_content,
            use_lint=use_lint,
            continuous=continuous_correctness,
        )
        correctness_mode = "syntax_lint"

    # Step 5: Combine
    reward = alpha * correctness_score + (1.0 - alpha) * sim_score

    meta = {
        **sim_meta,
        **correctness_meta,
        "sim_score": sim_score,
        "correctness_score": correctness_score,
        "correctness_mode": correctness_mode,
        "alpha": alpha,
        "combined_reward": reward,
    }
    return reward, meta


def calculate_rewards_batch(
    code_context: dict,
    oracle_new_content: dict,
    outputs: list,
    alpha: float = DEFAULT_ALPHA,
    continuous_correctness: bool = True,
    use_matcher_correctness: bool = True,
) -> tuple:
    """Compute combined rewards for a batch of G rollouts (one problem)."""
    rewards, metas = [], []
    for output in outputs:
        r, m = calculate_combined_reward(
            code_context,
            oracle_new_content,
            output,
            alpha,
            continuous_correctness=continuous_correctness,
            use_matcher_correctness=use_matcher_correctness,
        )
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

    Reward can be:
    - Discrete: {-1, 0, sim_score} where correctness is 0 or 1
    - Continuous (recommended): {-1} ∪ [0, 1] where correctness ∈ [0, 1]
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        use_lint: bool = True,
        continuous_correctness: bool = True,
        use_matcher_correctness: bool = True,
    ):
        self.alpha = alpha
        self.use_lint = use_lint
        self.continuous_correctness = continuous_correctness
        self.use_matcher_correctness = use_matcher_correctness

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
                    continuous_correctness=self.continuous_correctness,
                    use_matcher_correctness=self.use_matcher_correctness,
                )
            except Exception as e:
                logger.warning("Reward failed for completion %s: %s", idx, e)
                reward = FORMAT_PENALTY
            rewards.append(reward)

        return rewards
