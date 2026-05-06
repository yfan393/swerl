#!/usr/bin/env python
"""Quick test of patch extraction on one record."""

import json
import logging
import sys

# Setup logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

from data.extract_triples import extract_files_from_patch, extract_triple

# Load first record from filtered PRs
print("Loading first record from filtered_prs.jsonl...")
with open("data/raw/filtered_prs.jsonl") as f:
    record = json.loads(f.readline())

print(f"\nRecord: {record.get('instance_id')}")
print(f"Repo: {record.get('repo')}")
print(f"Python files expected: {record.get('python_files', [])}")
print(f"Patch size: {len(record.get('oracle_patch', ''))} chars")

# Test patch extraction
print("\n" + "="*80)
print("Testing patch extraction...")
print("="*80)
patch = record.get("oracle_patch", "")
print(f"First 500 chars of patch:\n{patch[:500]}")

files = extract_files_from_patch(patch)
print(f"\nExtracted files: {list(files.keys())}")
for fname, content in files.items():
    print(f"  {fname}: {len(content)} chars, {len(content.splitlines())} lines")

# Test full extraction
print("\n" + "="*80)
print("Testing full triple extraction...")
print("="*80)

# Check extracted content
print(f"\nExtracted content preview:")
for fname, content in files.items():
    print(f"\n{fname}:")
    print(f"  Length: {len(content)} chars")
    print(f"  First 200 chars:\n{content[:200]}")
    print(f"  Last 200 chars:\n{content[-200:]}")

# Try full extraction with detailed logging
logging.getLogger("__main__").setLevel(logging.DEBUG)
triple = extract_triple(record, "data/repos")
if triple:
    print(f"\nSUCCESS! Extracted triple for {triple['instance_id']}")
    print(f"  Files in code_context: {len(triple['file_contents'])} files")
    print(f"  Files with changes: {len(triple['oracle_new_content'])} files")
    if triple['oracle_new_content']:
        print(f"  Changed files: {list(triple['oracle_new_content'].keys())}")
else:
    print("\nFAILED to extract triple - check logs above")
