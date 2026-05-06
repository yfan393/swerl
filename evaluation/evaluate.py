"""
evaluation/evaluate.py
======================
Post-processing and evaluation after run_inference.py.

Steps:
  1. Parse raw_outputs.jsonl → extract best patch per instance via majority voting
  2. Write SWE-bench submission file (all_preds.jsonl)
  3. Compute format accuracy across the dataset
  4. (Optional) Compute sequence-similarity-based "oracle repair" score
     using the validation set ground truths (quick offline metric)

The SWE-bench resolve rate (pass@1) is computed by the official SWE-bench
evaluator after submission. This script prepares the submission file.

Two offline reward modes are available:
  - combined  : α·correctness + (1-α)·sim  (requires file_contents + oracle_new_content)
  - unidiff   : pure SequenceMatcher on raw unified diffs (lighter; uses upstream's
                calculate_reward_unidiff when available)

Usage:
    # Generate submission file
    python -m evaluation.evaluate \\
        --raw_output_dir outputs/eval/grpo \\
        --output_file outputs/eval/grpo/all_preds.jsonl

    # Offline combined reward on val set
    python -m evaluation.evaluate \\
        --mode val_reward \\
        --raw_output_dir outputs/eval/grpo \\
        --val_file data/processed/val.jsonl

    # Offline unidiff reward (lighter, no file I/O)
    python -m evaluation.evaluate \\
        --mode val_reward \\
        --reward_mode unidiff \\
        --raw_output_dir outputs/eval/grpo \\
        --val_file data/processed/val.jsonl
"""

import argparse
import difflib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# ── Import reward utilities ────────────────────────────────────────────────────
from reward.reward_fn import (
    calculate_combined_reward,
    apply_code_change,
    extract_thought_solution,
    parse_search_replace,
    FormatError,
)

# Optional: upstream calculate_reward_unidiff (lighter offline metric).
# Falls back to a local difflib implementation if swerl is not installed.
try:
    _SWERL_SRC = Path(__file__).parents[2] / "swe-rl-main" / "src"
    if _SWERL_SRC.exists() and str(_SWERL_SRC) not in sys.path:
        sys.path.insert(0, str(_SWERL_SRC))
    from swerl.core.reward import calculate_reward_unidiff as _upstream_unidiff
    _HAS_UNIDIFF = True
except ImportError:
    _HAS_UNIDIFF = False

# parse_thinking_output: extract <solution> block from model output
from utils.api_client import parse_thinking_output
from utils.io_utils import ensure_parent_dir, read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ════════════════════════════════════════════════════════════════════════════
# FORMAT VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def is_valid_format_check(output: str) -> bool:
    """
    Check whether a model output has a parseable SEARCH/REPLACE answer.

    Primary path  : extract_thought_solution → parse_search_replace (full format).
    Secondary path: parse_thinking_output → parse_search_replace (solution-only).
    """
    # Primary: require full <think>…</think><solution>…</solution> wrapping
    try:
        _, answer = extract_thought_solution(output)
        sr = parse_search_replace(answer)
        if sr:
            return True
    except FormatError:
        pass

    # Secondary: model emits only <solution>…</solution> (no <think> block)
    try:
        answer = parse_thinking_output(output)
        if answer and answer != output:
            sr = parse_search_replace(answer)
            return bool(sr)
    except Exception:
        pass

    return False


def _local_unidiff_reward(oracle_patch: str, output: str) -> float:
    """
    Fallback unidiff reward when swerl is not installed.

    Extracts the answer block from the model output and computes
    SequenceMatcher ratio against the oracle patch string.
    Returns -1.0 on format error.
    """
    import difflib

    try:
        _, answer = extract_thought_solution(output)
    except FormatError:
        answer = parse_thinking_output(output)
        if not answer or answer == output:
            return -1.0

    if not answer.strip():
        return -1.0

    return difflib.SequenceMatcher(
        None, oracle_patch.strip(), answer.strip(), autojunk=False
    ).ratio()


def calculate_reward_unidiff(oracle_patch: str, output: str) -> float:
    """
    Sequence-similarity reward between the oracle diff and the model answer block.

    Uses upstream swerl.core.reward.calculate_reward_unidiff when available,
    otherwise falls back to the local difflib implementation.

    Returns reward ∈ [-1, 1]  (-1 = format error).
    """
    if _HAS_UNIDIFF:
        try:
            reward, _ = _upstream_unidiff(oracle_patch=oracle_patch, output=output)
            return reward
        except Exception:
            pass
    return _local_unidiff_reward(oracle_patch, output)


def parse_patch_from_output(output: str) -> Optional[str]:
    """
    Extract the SEARCH/REPLACE answer block from a model output.

    Returns the raw SEARCH/REPLACE answer block, or None if parsing fails.
    generate_submission_file converts this answer into a unified diff before
    writing SWE-bench predictions.
    """
    # Try full format first
    try:
        _, answer = extract_thought_solution(output)
        sr = parse_search_replace(answer)
        if sr:
            return answer
    except FormatError:
        pass

    # Try solution-only format
    try:
        answer = parse_thinking_output(output)
        if answer and answer != output:
            sr = parse_search_replace(answer)
            if sr:
                return answer
    except Exception:
        pass

    return None


def search_replace_to_unified_diff(answer: str, file_contents: dict[str, str]) -> Optional[str]:
    """
    Convert SEARCH/REPLACE blocks to a unified diff using the saved eval files.

    SWE-bench submissions expect unified diffs, not raw SEARCH/REPLACE blocks.
    Returns None when the answer cannot be applied to the saved context.
    """
    sr = parse_search_replace(answer)
    if not sr:
        return None

    try:
        new_contents = apply_code_change(file_contents, sr)
    except Exception as e:
        logger.debug("SEARCH/REPLACE application failed during diff conversion: %s", e)
        return None

    diff_parts = []
    for path, new_content in new_contents.items():
        old_content = file_contents.get(path)
        if old_content is None or old_content == new_content:
            continue
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
        diff_text = "".join(diff)
        if diff_text:
            diff_parts.append(diff_text)

    if not diff_parts:
        return None
    return "\n".join(part.rstrip() for part in diff_parts) + "\n"


def parse_unified_patch_from_output(
    output: str,
    file_contents: dict[str, str],
) -> Optional[str]:
    answer = parse_patch_from_output(output)
    if answer is None:
        return None
    return search_replace_to_unified_diff(answer, file_contents)


def majority_vote_unified_diff(
    outputs: list[str],
    file_contents: dict[str, str],
) -> Optional[str]:
    """
    Majority vote over converted unified diffs.

    Voting after conversion avoids selecting a SEARCH/REPLACE answer that parses
    but cannot be applied to the saved file context.
    """
    valid = []
    for output in outputs:
        patch = parse_unified_patch_from_output(output, file_contents)
        if patch:
            valid.append(patch)

    if not valid:
        return None

    return Counter(p.strip() for p in valid).most_common(1)[0][0] + "\n"


def generate_submission_file(
    raw_output_dir: str,
    output_file: str,
    model_name: str = "llama3-swerl-8b",
):
    """
    Read raw_outputs.jsonl and produce all_preds.jsonl in SWE-bench format.

    SWE-bench expects:
        {"model_name_or_path": ..., "instance_id": ..., "model_patch": ...}
    """
    raw_file = Path(raw_output_dir) / "raw_outputs.jsonl"
    if not raw_file.exists():
        raise FileNotFoundError(f"raw_outputs.jsonl not found in {raw_output_dir}")

    records = read_jsonl(raw_file)

    logger.info(f"Processing {len(records)} instances for submission")

    total_valid = 0
    ensure_parent_dir(output_file)

    with open(output_file, "w") as out_f:
        for record in tqdm(records, desc="Building submission"):
            instance_id = record["instance_id"]
            outputs = record.get("outputs", [])

            file_contents = record.get("file_contents", {})
            model_patch = majority_vote_unified_diff(outputs, file_contents) or ""

            if model_patch:
                total_valid += 1
            else:
                if not file_contents:
                    logger.warning(
                        "%s has no saved file_contents; cannot build unified diff",
                        instance_id,
                    )

            submission = {
                "model_name_or_path": model_name,
                "instance_id": instance_id,
                "model_patch": model_patch,
            }
            out_f.write(json.dumps(submission) + "\n")

    pct = total_valid / len(records) if records else 0.0
    logger.info(
        f"Submission written to {output_file}\n"
        f"  Instances with valid patch: {total_valid}/{len(records)} ({pct:.1%})"
    )


def offline_val_reward(
    raw_output_dir: str,
    val_file: str,
    num_samples: int = 20,
    alpha: float = 0.3,
    reward_mode: str = "combined",
) -> dict:
    """
    Compute offline reward on the validation set.

    reward_mode options:
      "combined" : α·correctness + (1-α)·similarity
                   (requires file_contents + oracle_new_content in val records)
      "unidiff"  : pure SequenceMatcher on oracle_patch string vs. answer block
                   (lighter — only needs oracle_patch field)

    Returns a dict with aggregated statistics.
    """
    raw_file = Path(raw_output_dir) / "raw_outputs.jsonl"
    if not raw_file.exists():
        raise FileNotFoundError(f"raw_outputs.jsonl not found in {raw_output_dir}")

    # Load val records
    val_records = {rec["instance_id"]: rec for rec in read_jsonl(val_file)}

    raw_outputs = {
        rec["instance_id"]: rec.get("outputs", [])
        for rec in read_jsonl(raw_file)
    }

    logger.info(
        f"Offline val reward ({reward_mode} mode) over "
        f"{len(val_records)} val instances, up to {num_samples} samples each"
    )

    rewards_per_instance: dict[str, dict] = {}
    total_format_correct = 0
    total_evaluated = 0

    for instance_id, record in tqdm(val_records.items(), desc="Computing val rewards"):
        outputs = raw_outputs.get(instance_id, [])
        if not outputs:
            continue

        sample = outputs[:num_samples]

        if reward_mode == "combined":
            file_contents = record.get("file_contents", {})
            oracle_new_content = record.get("oracle_new_content", {})
            if not file_contents or not oracle_new_content:
                logger.debug(f"Skipping {instance_id}: missing file_contents/oracle_new_content")
                continue

            rewards = []
            for output in sample:
                r, _ = calculate_combined_reward(
                    code_context=file_contents,
                    oracle_new_content=oracle_new_content,
                    output=output,
                    alpha=alpha,
                )
                rewards.append(r)
                # Format-valid outputs have reward >= 0; negatives are format errors.
                if r >= 0:
                    total_format_correct += 1
                total_evaluated += 1

        elif reward_mode == "unidiff":
            oracle_patch = record.get("oracle_patch", "")
            if not oracle_patch:
                logger.debug(f"Skipping {instance_id}: no oracle_patch")
                continue

            rewards = []
            for output in sample:
                r = calculate_reward_unidiff(oracle_patch=oracle_patch, output=output)
                rewards.append(r)
                if r >= 0:
                    total_format_correct += 1
                total_evaluated += 1

        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode!r}")

        if not rewards:
            continue

        rewards_per_instance[instance_id] = {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "format_rate": sum(1 for r in rewards if r >= 0) / len(rewards),
        }

    all_means = [v["mean_reward"] for v in rewards_per_instance.values()]
    all_maxes = [v["max_reward"] for v in rewards_per_instance.values()]

    results = {
        "num_instances": len(rewards_per_instance),
        "reward_mode": reward_mode,
        "alpha": alpha if reward_mode == "combined" else None,
        "mean_reward_across_instances": sum(all_means) / len(all_means) if all_means else 0.0,
        "mean_max_reward": sum(all_maxes) / len(all_maxes) if all_maxes else 0.0,
        "format_accuracy": total_format_correct / total_evaluated if total_evaluated > 0 else 0.0,
        "per_instance": rewards_per_instance,
    }

    report_file = Path(raw_output_dir) / f"val_reward_{reward_mode}_report.json"
    with open(report_file, "w") as f:
        # Omit per_instance from the summary file to keep it readable
        json.dump({k: v for k, v in results.items() if k != "per_instance"}, f, indent=2)

    logger.info(
        f"Validation reward ({reward_mode}):\n"
        f"  Instances evaluated : {results['num_instances']}\n"
        f"  Mean reward         : {results['mean_reward_across_instances']:.4f}\n"
        f"  Mean max reward     : {results['mean_max_reward']:.4f}\n"
        f"  Format accuracy     : {results['format_accuracy']:.1%}"
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["submission", "val_reward"], default="submission",
        help="'submission' builds all_preds.jsonl; 'val_reward' runs offline eval"
    )
    parser.add_argument("--raw_output_dir", required=True)
    parser.add_argument("--output_file", default=None, help="Submission output path")
    parser.add_argument("--val_file", default="data/processed/val.jsonl")
    parser.add_argument("--model_name", default="llama3-swerl-8b")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Samples to evaluate per val instance")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Correctness weight α for combined reward")
    parser.add_argument(
        "--reward_mode", choices=["combined", "unidiff"], default="combined",
        help="Offline reward metric: 'combined' (full) or 'unidiff' (lightweight)"
    )
    args = parser.parse_args()

    if args.mode == "submission":
        output_file = args.output_file or str(Path(args.raw_output_dir) / "all_preds.jsonl")
        generate_submission_file(
            raw_output_dir=args.raw_output_dir,
            output_file=output_file,
            model_name=args.model_name,
        )
    elif args.mode == "val_reward":
        offline_val_reward(
            raw_output_dir=args.raw_output_dir,
            val_file=args.val_file,
            num_samples=args.num_samples,
            alpha=args.alpha,
            reward_mode=args.reward_mode,
        )


if __name__ == "__main__":
    main()
