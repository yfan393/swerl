"""
extract_triples.py
==================
Converts filtered PR records into training triples:
    (problem_statement, code_context, oracle_patch)

For each filtered PR:
  1. Clone (or fetch) the base commit of the repo
  2. For each Python file changed in the PR, read the pre-patch content
  3. Build code_context by concatenating file contents (respecting token budget)
  4. Store the oracle_patch (the actual diff)
  5. Select a random 10k subset and split train/val

Also computes oracle_new_content (post-patch file contents) needed by the reward function.

Usage:
    python -m data.extract_triples \
        --input_file data/raw/filtered_prs.jsonl \
        --output_dir data/processed \
        --repo_cache_dir data/repos \
        --num_seeds 10000
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Use upstream git utilities for robust patch application
from utils.git_utils import (
    fake_git_apply_multiple,
    check_syntax,
    check_code_differ_by_just_empty_lines,
)
from utils.io_utils import read_jsonl, write_jsonl
from utils.repo_utils import ensure_repo_cloned, read_file_at_commit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_CONTEXT_TOKENS = 12_000   # Rough char-based budget (≈ 3 chars/token)
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 3
PLAYGROUND_DIR = os.getenv("PLAYGROUND_DIR", "playground")


def build_full_code_context(file_contents: dict[str, str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Concatenate file contents into a single context string, respecting the char budget.
    Files are truncated rather than dropped if they are very large.
    """
    parts = []
    used = 0
    for path, content in file_contents.items():
        header = f"### {path}\n"
        chunk = header + content
        if used + len(chunk) > max_chars:
            remaining = max_chars - used - len(header) - 100
            if remaining > 200:
                chunk = header + content[:remaining] + "\n# ... (truncated)"
            else:
                break
        parts.append(chunk)
        used += len(chunk)

    return "\n\n".join(parts)


def extract_triple(
    record: dict,
    repo_cache_dir: str,
) -> Optional[dict]:
    """
    For a single filtered PR record, build the (issue, code_ctx, oracle_patch) triple.
    Also computes oracle_new_content for reward calculation.
    """
    repo = record["repo"]
    base_sha = record.get("base_sha", "")
    python_files = record.get("python_files", [])
    oracle_patch = record.get("oracle_patch", "")

    if not base_sha or not python_files or not oracle_patch:
        return None

    # Ensure repo is cloned
    try:
        repo_path = ensure_repo_cloned(repo, repo_cache_dir)
    except RuntimeError as e:
        logger.warning("%s", e)
        return None

    # Read each Python file at base commit
    file_contents: dict[str, str] = {}
    for fpath in python_files:
        content = read_file_at_commit(repo_path, base_sha, fpath)
        if content is not None:
            file_contents[fpath] = content

    if not file_contents:
        return None

    # ── Apply the oracle patch via `git apply` (robust to whitespace quirks) ──
    # fake_git_apply_multiple() creates a temp git repo, commits the original
    # files, then runs `git apply` on the full PR diff.  This is the same
    # approach the original swe-rl-main uses for normalising patches.
    os.makedirs(PLAYGROUND_DIR, exist_ok=True)
    try:
        patched_contents = fake_git_apply_multiple(
            repo_playground=PLAYGROUND_DIR,
            file_path_contents=file_contents,
            patch=oracle_patch,
        )
    except Exception as e:
        logger.debug(f"fake_git_apply_multiple failed for {record.get('instance_id')}: {e}")
        return None

    # Only keep files that actually changed and have valid Python syntax
    oracle_new_content: dict[str, str] = {}
    for fpath, new_content in patched_contents.items():
        original = file_contents.get(fpath, "")
        if new_content == original:
            continue   # patch didn't touch this file
        if not check_syntax(new_content):
            continue   # broken output — skip
        if check_code_differ_by_just_empty_lines([new_content], [original]):
            continue   # trivial whitespace-only change — not useful for training
        oracle_new_content[fpath] = new_content

    if not oracle_new_content:
        # Patch applied but produced no meaningful Python changes — skip
        return None

    code_context = build_full_code_context(file_contents)

    return {
        "instance_id": record["instance_id"],
        "repo": repo,
        "pr_number": record["pr_number"],
        "issue_number": record.get("issue_number"),
        "problem_statement": record["problem_statement"],
        "code_context": code_context,
        "file_contents": file_contents,          # path -> original content
        "oracle_new_content": oracle_new_content, # path -> patched content
        "oracle_patch": oracle_patch,
        "python_files": python_files,
        "merged_at": record.get("merged_at", ""),
    }


def extract_triples(
    input_file: str,
    output_dir: str,
    repo_cache_dir: str,
    num_seeds: int = 10_000,
    train_ratio: float = 0.95,
    seed: int = 42,
):
    """Main extraction pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_file)

    logger.info(f"Loaded {len(records)} filtered PR records")

    # Shuffle and process
    random.seed(seed)
    random.shuffle(records)

    triples = []
    failed = 0
    pbar = tqdm(records, desc="Extracting triples")

    for record in pbar:
        if len(triples) >= num_seeds:
            break

        triple = extract_triple(record, repo_cache_dir)
        if triple is not None:
            triples.append(triple)
        else:
            failed += 1

        pbar.set_postfix({"ok": len(triples), "fail": failed})

    logger.info(f"Extracted {len(triples)} triples ({failed} failed)")

    # Train/val split
    split_idx = int(len(triples) * train_ratio)
    train_triples = triples[:split_idx]
    val_triples = triples[split_idx:]

    for split_name, split_data in [("train", train_triples), ("val", val_triples)]:
        out_file = output_path / f"{split_name}.jsonl"
        write_jsonl(split_data, out_file)  # Fixed: records first, file_path second
        logger.info(f"Wrote {len(split_data)} records to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/raw/filtered_prs.jsonl")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--repo_cache_dir", default="data/repos")
    parser.add_argument("--num_seeds", type=int, default=10_000)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_triples(
        input_file=args.input_file,
        output_dir=args.output_dir,
        repo_cache_dir=args.repo_cache_dir,
        num_seeds=args.num_seeds,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
