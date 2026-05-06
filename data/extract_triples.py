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
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Use upstream git utilities for robust patch application
from utils.git_utils import (
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


def apply_unified_diff(original_files: dict[str, str], patch_text: str) -> dict[str, str]:
    """
    Apply unified diff to files using git apply.

    This is more robust than manual parsing as it handles:
    - Context lines, added lines, removed lines
    - Binary files
    - Renames, deletes, creates
    - Edge cases in diff format

    Args:
        original_files: Dict of {filepath: content}
        patch_text: Unified diff text

    Returns:
        Dict of {filepath: patched_content}

    Raises:
        Exception if patch cannot be applied
    """
    import shutil

    # Create temp directory for git repo
    temp_dir = tempfile.mkdtemp(prefix="swerl_diff_")
    try:
        # Initialize bare git repo
        subprocess.run(
            ["git", "init"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
        )

        # Configure git user
        subprocess.run(
            ["git", "config", "user.email", "swerl@example.com"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "SWE-RL"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
        )

        # Write original files
        for filepath, content in original_files.items():
            file_path = Path(temp_dir) / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        # Commit original files
        subprocess.run(
            ["git", "add", "-A"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "original"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
        )

        # Apply patch using git apply
        logger.debug(f"Applying patch ({len(patch_text)} bytes) in temp dir")

        # Try standard git apply first
        result = subprocess.run(
            ["git", "apply"],
            input=patch_text.encode("utf-8"),
            cwd=temp_dir,
            capture_output=True,
        )

        # If standard apply fails, try with 3-way merge (more lenient)
        if result.returncode != 0:
            logger.debug(f"Standard git apply failed, trying 3-way merge")
            result = subprocess.run(
                ["git", "apply", "--3way"],
                input=patch_text.encode("utf-8"),
                cwd=temp_dir,
                capture_output=True,
            )

        # If 3-way fails, try with reduced context requirements
        if result.returncode != 0:
            logger.debug(f"3-way merge failed, trying with reduced context (-C 1)")
            result = subprocess.run(
                ["git", "apply", "-C1"],
                input=patch_text.encode("utf-8"),
                cwd=temp_dir,
                capture_output=True,
            )

        # If still failing, try with unidiff-zero (for zero-context patches)
        if result.returncode != 0:
            logger.debug(f"Reduced context failed, trying with unidiff-zero")
            result = subprocess.run(
                ["git", "apply", "--unidiff-zero"],
                input=patch_text.encode("utf-8"),
                cwd=temp_dir,
                capture_output=True,
            )

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="ignore")
            logger.warning(f"All patch application methods failed for {len(patch_text)} byte patch")
            logger.debug(f"stderr: {error_msg[:500]}")
            raise RuntimeError(f"git apply failed: {error_msg}")

        # Determine which method worked
        if result.returncode == 0:
            # Check which method succeeded by checking if any changes were made
            patched_files = {}
            for filepath in original_files:
                file_path = Path(temp_dir) / filepath
                if file_path.exists():
                    patched_files[filepath] = file_path.read_text(encoding="utf-8")
                else:
                    # File was deleted by patch
                    patched_files[filepath] = ""
            return patched_files
        else:
            raise RuntimeError(f"All patch application methods failed")

    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_files_from_patch(patch: str) -> dict[str, str]:
    """
    Extract pre-patch file contents directly from unified diff format.

    Returns dict mapping file_path -> original content.
    This reconstructs the original file by:
    - Including all context lines (prefix: space)
    - Including all removed lines (prefix: -)
    - Excluding all added lines (prefix: +)
    """
    files = {}
    current_file = None
    current_content = []
    in_file_diff = False
    diff_headers_found = 0

    for line in patch.split('\n'):
        # Match: diff --git a/path/to/file b/path/to/file
        if line.startswith('diff --git'):
            diff_headers_found += 1
            # Save previous file if any
            if current_file is not None and current_content:
                # Remove trailing empty lines from reconstruction
                while current_content and current_content[-1] == '':
                    current_content.pop()
                files[current_file] = '\n'.join(current_content)
                logger.debug(f"Extracted {current_file}: {len(current_content)} lines, {len(''.join(current_content))} bytes")

            # Extract path: "a/path/to/file" -> "path/to/file"
            match = re.search(r'^diff --git a/(.+?) b/.+?$', line)
            if match:
                current_file = match.group(1)
                current_content = []
                in_file_diff = True
                logger.debug(f"Found file #{diff_headers_found}: {current_file}")
            else:
                logger.warning(f"Could not parse diff header: {line[:80]}")
                current_file = None
                in_file_diff = False

        # Skip git metadata lines (---, +++, index, new file, deleted file, etc.)
        elif line.startswith(('index ', 'new file', 'deleted file', 'similarity', 'rename')):
            continue

        # Skip "--- " and "+++ " file headers
        elif line.startswith('---') or line.startswith('+++'):
            continue

        # Skip diff hunk headers (@@)
        elif line.startswith('@@'):
            continue

        # Skip "\ No newline at end of file"
        elif line.startswith('\\'):
            continue

        # Process diff content lines
        elif in_file_diff and current_file is not None:
            if len(line) == 0:
                # Empty line - part of the file
                current_content.append('')
            elif line[0] == ' ':
                # Context line - part of original file, strip leading space
                current_content.append(line[1:])
            elif line[0] == '-':
                # Removed line - part of original file, strip leading minus
                current_content.append(line[1:])
            elif line[0] == '+':
                # Added line - skip, not part of original
                pass
            # else: shouldn't happen in well-formed diff

    # Don't forget the last file
    if current_file is not None and current_content:
        while current_content and current_content[-1] == '':
            current_content.pop()
        files[current_file] = '\n'.join(current_content)
        logger.debug(f"Extracted {current_file}: {len(current_content)} lines, {len(''.join(current_content))} bytes")

    logger.info(f"Extracted {len(files)} files from patch with {diff_headers_found} diff headers found")
    for fpath, content in files.items():
        logger.debug(f"  {fpath}: {len(content)} bytes, {len(content.splitlines())} lines")
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
    logger.debug(f"Extracting files from patch for {instance_id} ({len(oracle_patch)} chars)")
    file_contents = extract_files_from_patch(oracle_patch)
    logger.info(f"Extracted {len(file_contents)} files for {instance_id}: {list(file_contents.keys())}")

    if not file_contents:
        logger.error(f"No files extracted from patch for {instance_id}")
        return None

    # ── Apply the oracle patch via `git apply` (robust to whitespace quirks) ──
    # Creates a temp git repo, commits the original files,
    # then runs `git apply` on the full PR diff.
    try:
        patched_contents = apply_unified_diff(file_contents, oracle_patch)
        logger.info(f"Successfully applied patch for {instance_id}: {len(patched_contents)} files patched")
    except Exception as e:
        logger.warning(f"Patch application failed for {instance_id}: {e}")
        return None

    # Only keep files that actually changed and have valid Python syntax
    oracle_new_content: dict[str, str] = {}
    skipped_reasons = {"unchanged": 0, "syntax_error": 0, "whitespace_only": 0}

    for fpath, new_content in patched_contents.items():
        original = file_contents.get(fpath, "")
        if new_content == original:
            logger.debug(f"Skipping {fpath}: content unchanged after patch")
            skipped_reasons["unchanged"] += 1
            continue   # patch didn't touch this file
        if not check_syntax(new_content):
            logger.debug(f"Skipping {fpath}: broken syntax after patch")
            logger.debug(f"  Original syntax valid: {check_syntax(original)}")
            skipped_reasons["syntax_error"] += 1
            continue   # broken output — skip
        if check_code_differ_by_just_empty_lines(new_content, original):
            logger.debug(f"Skipping {fpath}: only whitespace changes")
            skipped_reasons["whitespace_only"] += 1
            continue   # trivial whitespace-only change — not useful for training
        oracle_new_content[fpath] = new_content
        logger.info(f"Keeping {fpath}: {len(new_content)} chars, {len(original)} chars original")

    if not oracle_new_content:
        # Patch applied but produced no meaningful Python changes — skip
        logger.warning(f"No meaningful changes for {instance_id}: {skipped_reasons}")
        return None

    logger.info(f"Extracted {len(oracle_new_content)} meaningful files for {instance_id}")

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
