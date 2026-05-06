#!/usr/bin/env python3
"""
Debug script to examine patch extraction issues.
"""

import json
import logging
from pathlib import Path
from data.extract_triples import extract_files_from_patch
from utils.io_utils import read_jsonl

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load first record
    records = read_jsonl("data/raw/filtered_prs.jsonl")
    if not records:
        logger.error("No records found!")
        return

    record = records[0]
    instance_id = record.get("instance_id", "unknown")
    oracle_patch = record.get("oracle_patch", "")

    logger.info(f"\n{'='*70}")
    logger.info(f"Instance: {instance_id}")
    logger.info(f"Oracle patch size: {len(oracle_patch)} bytes")
    logger.info(f"{'='*70}\n")

    # Show patch header
    lines = oracle_patch.split('\n')
    logger.info("Patch first 20 lines:")
    for i, line in enumerate(lines[:20]):
        print(f"{i+1:3d}: {line[:100]}")

    # Extract files
    logger.info(f"\n{'='*70}")
    logger.info("Extracting files from patch...")
    logger.info(f"{'='*70}\n")

    files = extract_files_from_patch(oracle_patch)

    logger.info(f"\nExtracted {len(files)} files:")
    for fpath, content in files.items():
        logger.info(f"\n  File: {fpath}")
        logger.info(f"    Size: {len(content)} bytes, {len(content.splitlines())} lines")
        logger.info(f"    First 5 lines:")
        for line in content.splitlines()[:5]:
            logger.info(f"      {line[:80]}")
        logger.info(f"    Last 5 lines:")
        for line in content.splitlines()[-5:]:
            logger.info(f"      {line[:80]}")

    # Show patch structure
    logger.info(f"\n{'='*70}")
    logger.info("Patch structure analysis:")
    logger.info(f"{'='*70}\n")

    diff_sections = [line for line in lines if line.startswith('diff --git')]
    logger.info(f"Total diff sections: {len(diff_sections)}")
    for i, line in enumerate(diff_sections, 1):
        logger.info(f"  {i}. {line}")

    # Check for hunk headers
    hunk_count = len([line for line in lines if line.startswith('@@')])
    logger.info(f"\nTotal hunks: {hunk_count}")

    # Check for removed vs added lines
    removed = len([line for line in lines if line.startswith('-') and not line.startswith('---')])
    added = len([line for line in lines if line.startswith('+') and not line.startswith('+++')])
    context = len([line for line in lines if line.startswith(' ')])

    logger.info(f"\nLine types in patch:")
    logger.info(f"  Context lines (prefix: space): {context}")
    logger.info(f"  Removed lines (prefix: -): {removed}")
    logger.info(f"  Added lines (prefix: +): {added}")

    # Try applying patch manually
    logger.info(f"\n{'='*70}")
    logger.info("Attempting to apply patch...")
    logger.info(f"{'='*70}\n")

    import subprocess
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="patch_debug_")
    try:
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_dir, capture_output=True, check=True)

        # Write files
        for fpath, content in files.items():
            file_path = Path(temp_dir) / fpath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"Wrote: {fpath}")

        # Commit
        subprocess.run(["git", "add", "-A"], cwd=temp_dir, capture_output=True, check=True)
        result = subprocess.run(["git", "commit", "-m", "initial"], cwd=temp_dir, capture_output=True)
        if result.returncode == 0:
            logger.info("Committed initial files")
        else:
            logger.warning(f"Commit failed: {result.stderr.decode()[:200]}")

        # Try patch
        logger.info("\nApplying patch with git apply...")
        result = subprocess.run(
            ["git", "apply"],
            input=oracle_patch.encode("utf-8"),
            cwd=temp_dir,
            capture_output=True,
        )

        if result.returncode == 0:
            logger.info("✓ Patch applied successfully!")
        else:
            logger.error("✗ Patch application failed!")
            stderr = result.stderr.decode("utf-8", errors="ignore")
            logger.error(f"\nError output (first 1000 chars):\n{stderr[:1000]}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
