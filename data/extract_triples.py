"""
extract_triples.py
==================
Converts filtered PR records into training triples:
    (problem_statement, code_context, oracle_patch)

For each filtered PR:
  1. Extract pre-patch file contents directly from the oracle_patch (unified diff)
  2. Build code_context by concatenating file contents (respecting token budget)
  3. Apply patch to compute oracle_new_content (post-patch file contents)
  4. Validate patch application and syntax
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
import re
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


def extract_files_from_patch(patch: str) -> dict[str, str]:
    """
    Extract pre-patch file contents directly from unified diff format.

    Returns dict mapping file_path -> original content.
    This avoids git operations and directly parses the diff.
    """
    files = {}
    current_file = None
    current_content = []
    in_file_diff = False

    for line in patch.split('\n'):
        # Match: diff --git a/path/to/file b/path/to/file
        if line.startswith('diff --git'):
            # Save previous file if any
            if current_file is not None and current_content:
                files[current_file] = '\n'.join(current_content)

            # Extract path: "a/path/to/file" -> "path/to/file"
            match = re.search(r'^diff --git a/(.+?) b/.+?$', line)
            if match:
                current_file = match.group(1)
                current_content = []
                in_file_diff = True
            else:
                current_file = None
                in_file_diff = False

        # Skip git metadata lines (---, +++, index, new file, deleted file, etc.)
        elif line.startswith(('---', '+++', 'index ', 'new file', 'deleted file', 'similarity', 'rename')):
            continue

        # Skip diff hunk headers (@@)
        elif line.startswith('@@'):
            continue

        # Skip "\ No newline at end of file"
        elif line.startswith('\\'):
            continue

        # Context lines start with space - include content after the space
        elif in_file_diff and line.startswith(' ') and current_file is not None:
            # Strip the leading space to get original content
            current_content.append(line[1:])

        # Removed lines start with '-' - include content after the '-'
        # These are part of the original file (pre-patch)
        elif in_file_diff and line.startswith('-') and current_file is not None:
            # Only include if it's a removed line (not a diff header like "---")
            if not line.startswith('---'):
                current_content.append(line[1:])

        # Added lines start with '+' - skip these, they're post-patch
        elif in_file_diff and line.startswith('+'):
            if not line.startswith('+++'):
                # This is new content added by the patch, skip it
                pass

        # Empty lines (just a newline)
        elif in_file_diff and line == '' and current_file is not None:
            current_content.append('')

    # Don't forget the last file
    if current_file is not None and current_content:
        files[current_file] = '\n'.join(current_content)

    return files


def extract_triple(
    record: dict,
    repo_cache_dir: str,
) -> Optional[dict]:
    """
    For a single filtered PR record, build the (issue, code_ctx, oracle_patch) triple.
    Also computes oracle_new_content for reward calculation.

    Extracts pre-patch file contents directly from the oracle_patch (unified diff),
    avoiding git operations entirely.
    """
    instance_id = record.get("instance_id")
    repo = record.get("repo", "")
    oracle_patch = record.get("oracle_patch", "")
    problem_statement = record.get("problem_statement", "")

    if not oracle_patch or not problem_statement:
        logger.warning(f"Record {instance_id} missing oracle_patch or problem_statement")
        return None

    # Extract file contents directly from the patch
    # This avoids git operations and uses the patch as the source of truth
    file_contents = extract_files_from_patch(oracle_patch)

    if not file_contents:
        logger.warning(f"No files extracted from patch for {instance_id}")
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
        logger.debug(f"fake_git_apply_multiple failed for {instance_id}: {e}")
        return None

    # Only keep files that actually changed and have valid Python syntax
    oracle_new_content: dict[str, str] = {}
    for fpath, new_content in patched_contents.items():
        original = file_contents.get(fpath, "")
        if new_content == original:
            continue   # patch didn't touch this file
        if not check_syntax(new_content):
            logger.debug(f"Skipping {fpath} in {instance_id}: broken syntax after patch")
            continue   # broken output — skip
        if check_code_differ_by_just_empty_lines([new_content], [original]):
            logger.debug(f"Skipping {fpath} in {instance_id}: only whitespace changes")
            continue   # trivial whitespace-only change — not useful for training
        oracle_new_content[fpath] = new_content

    if not oracle_new_content:
        # Patch applied but produced no meaningful Python changes — skip
        logger.debug(f"No meaningful changes for {instance_id} after patch application")
        return None

    code_context = build_full_code_context(file_contents)

    return {
        "instance_id": instance_id,
        "repo": repo,
        "pr_number": record.get("pr_number"),
        "issue_number": record.get("issue_number"),
        "problem_statement": problem_statement,
        "code_context": code_context,
        "file_contents": file_contents,          # path -> original content
        "oracle_new_content": oracle_new_content, # path -> patched content
        "oracle_patch": oracle_patch,
        "python_files": list(file_contents.keys()),  # Files actually extracted
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
