#!/usr/bin/env python3
"""
Test script to debug single record extraction.
"""

import sys
import logging
import json
from pathlib import Path

# Add swerl to path
sys.path.insert(0, str(Path(__file__).parent))

from data.extract_triples import extract_triple
from utils.io_utils import read_jsonl

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    # Read filtered PRs
    input_file = Path("data/raw/filtered_prs.jsonl")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    records = read_jsonl(str(input_file))
    logger.info(f"Loaded {len(records)} records")

    if not records:
        logger.error("No records loaded!")
        return

    # Test first record
    record = records[0]
    instance_id = record.get("instance_id", "unknown")
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing record: {instance_id}")
    logger.info(f"{'='*60}")

    # Print record metadata
    logger.info(f"Problem statement length: {len(record.get('problem_statement', ''))}")
    logger.info(f"Oracle patch length: {len(record.get('oracle_patch', ''))}")
    logger.info(f"Python files: {record.get('python_files', [])}")

    # Run extraction
    logger.info("\nStarting extraction...")
    result = extract_triple(record, repo_cache_dir="data/repos")

    if result:
        logger.info(f"\n✓ Extraction SUCCESSFUL")
        logger.info(f"  Instance ID: {result['instance_id']}")
        logger.info(f"  Code context length: {len(result['code_context'])}")
        logger.info(f"  Files in original: {len(result['file_contents'])}")
        logger.info(f"  Files in oracle_new_content: {len(result['oracle_new_content'])}")
        for fpath in result['oracle_new_content']:
            logger.info(f"    - {fpath}")
    else:
        logger.error(f"\n✗ Extraction FAILED")
        logger.error(f"  Check logs above for details")

if __name__ == "__main__":
    main()
