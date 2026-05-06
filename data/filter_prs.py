"""
filter_prs.py
=============
Secondary filtering pass using the GitHub REST API to:
  1. Fetch the actual PR diff and check it modifies at least one Python file
  2. Fetch the linked issue text (problem_statement)
  3. Remove bot-authored PRs
  4. Remove diffs that are too small/large, lock-file-only, or test-only
  5. Verify the issue is a bug report (not a feature request)

Input:  raw_prs.jsonl from fetch_gharchive.py
Output: filtered_prs.jsonl — richer records ready for extract_triples.py

Usage:
    python -m data.filter_prs \
        --input_file data/raw/raw_prs.jsonl \
        --output_file data/raw/filtered_prs.jsonl \
        --max_records 20000
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from utils.io_utils import load_jsonl_id_set, read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GITHUB_API = "https://api.github.com"

# Files we always ignore
SKIP_FILE_PATTERNS = re.compile(
    r"(\.lock$|requirements.*\.txt$|poetry\.lock$|package-lock\.json$"
    r"|yarn\.lock$|Pipfile\.lock$|setup\.cfg$|\.toml$)",
    re.IGNORECASE,
)

BOT_PATTERNS = re.compile(r"\[bot\]|dependabot|renovate|greenkeeper|snyk", re.IGNORECASE)

# Heuristic: issue body should mention a bug, not just a feature
BUG_ISSUE_PATTERN = re.compile(
    r"\b(bug|error|exception|crash|fail|broken|wrong|incorrect|unexpected|traceback|stacktrace)\b",
    re.IGNORECASE,
)

CLOSES_PATTERN = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE)


def get_github_session() -> requests.Session:
    """Build a requests Session with GitHub auth if token is set."""
    session = requests.Session()
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        session.headers["Authorization"] = f"token {token}"
    session.headers["Accept"] = "application/vnd.github.v3+json"
    session.headers["User-Agent"] = "swe-rl-reimpl/1.0"
    return session


def github_get(session: requests.Session, url: str, max_retries: int = 5) -> Optional[dict | list]:
    """GET request with exponential backoff on rate-limit / server errors."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 403:
                # Rate limited — wait until reset
                reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_ts - time.time() + 2, 5)
                logger.warning(f"Rate limited. Sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"Failed {url}: {e}")
                return None
    return None


def extract_linked_issue_numbers(pr_title: str, pr_body: str) -> list[int]:
    """Extract issue numbers referenced in PR title/body."""
    text = (pr_title or "") + " " + (pr_body or "")
    matches = CLOSES_PATTERN.findall(text)
    return [int(m) for m in matches]


def fetch_pr_details(
    session: requests.Session,
    repo: str,
    pr_number: int,
    max_diff_chars: int = 500_000,
    max_files_changed: int = 20,
    min_python_files: int = 1,
    min_diff_chars: int = 50,
    skip_patterns: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Fetch PR diff and files from GitHub API.
    Returns dict with 'files', 'diff_text' or None on failure.
    """
    # Get list of changed files
    files_url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
    files = github_get(session, files_url)
    if not files or not isinstance(files, list):
        return None

    # Collect Python files with their diffs
    python_files = []
    all_diff_text = []
    total_additions = 0
    total_deletions = 0

    for f in files:
        filename = f.get("filename", "")
        patch = f.get("patch", "")

        configured_skip = any(
            fnmatch.fnmatch(filename, pattern) for pattern in (skip_patterns or [])
        )
        if SKIP_FILE_PATTERNS.search(filename) or configured_skip:
            continue

        if filename.endswith(".py"):
            python_files.append(filename)
            if patch:
                all_diff_text.append(f"diff --git a/{filename} b/{filename}\n{patch}")
                total_additions += f.get("additions", 0)
                total_deletions += f.get("deletions", 0)

    if len(python_files) < min_python_files:
        return None

    if len(python_files) > max_files_changed:
        return None

    diff_text = "\n".join(all_diff_text)
    if len(diff_text) > max_diff_chars:
        return None

    if len(diff_text) < min_diff_chars:
        return None

    if total_additions + total_deletions < 3:
        return None  # Diff too small

    if total_additions + total_deletions > 5000:
        return None  # Diff too large

    return {
        "python_files": python_files,
        "diff_text": diff_text,
        "total_additions": total_additions,
        "total_deletions": total_deletions,
        "num_files": len(python_files),
    }


def fetch_issue_text(
    session: requests.Session,
    repo: str,
    issue_numbers: list[int],
) -> Optional[tuple[int, str]]:
    """
    Try to fetch the best matching issue. Returns (issue_number, body_text) or None.
    """
    for issue_num in issue_numbers:
        url = f"{GITHUB_API}/repos/{repo}/issues/{issue_num}"
        issue = github_get(session, url)
        if not issue or not isinstance(issue, dict):
            continue

        # Skip pull requests returned as issues
        if issue.get("pull_request"):
            continue

        title = issue.get("title", "")
        body = issue.get("body") or ""
        combined = title + " " + body

        # Ensure it looks like a bug report
        if not BUG_ISSUE_PATTERN.search(combined):
            continue

        return issue_num, f"## {title}\n\n{body}"

    return None


def process_pr(
    session: requests.Session,
    raw_pr: dict,
    max_diff_chars: int = 500_000,
    max_files_changed: int = 20,
    min_python_files: int = 1,
    min_diff_chars: int = 50,
    skip_patterns: Optional[list[str]] = None,
    bot_suffixes: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Full filtering pipeline for a single PR.
    Returns enriched dict or None if filtered out.
    """
    repo = raw_pr["repo"]
    pr_number = raw_pr["pr_number"]
    author = raw_pr.get("author", "")

    # Skip bots
    configured_bot = any(author.lower().endswith(s.lower()) for s in (bot_suffixes or []))
    if BOT_PATTERNS.search(author) or configured_bot:
        return None

    # Fetch PR diff & file list
    pr_details = fetch_pr_details(
        session,
        repo,
        pr_number,
        max_diff_chars=max_diff_chars,
        max_files_changed=max_files_changed,
        min_python_files=min_python_files,
        min_diff_chars=min_diff_chars,
        skip_patterns=skip_patterns,
    )
    if pr_details is None:
        return None

    # Find linked issue(s)
    issue_numbers = extract_linked_issue_numbers(
        raw_pr.get("pr_title", ""), raw_pr.get("pr_body", "")
    )
    if not issue_numbers:
        return None

    issue_result = fetch_issue_text(session, repo, issue_numbers)
    if issue_result is None:
        return None

    issue_number, problem_statement = issue_result

    # Build instance_id (mirrors SWE-bench convention)
    repo_slug = repo.replace("/", "__")
    instance_id = f"{repo_slug}-{pr_number}"

    return {
        "instance_id": instance_id,
        "repo": repo,
        "pr_number": pr_number,
        "issue_number": issue_number,
        "problem_statement": problem_statement,
        "oracle_patch": pr_details["diff_text"],
        "python_files": pr_details["python_files"],
        "base_sha": raw_pr.get("base_sha", ""),
        "head_sha": raw_pr.get("head_sha", ""),
        "merged_at": raw_pr.get("merged_at", ""),
        "html_url": raw_pr.get("html_url", ""),
    }


def filter_prs(
    input_file: str,
    output_file: str,
    max_records: Optional[int] = None,
    max_workers: int = 8,
    max_diff_chars: int = 500_000,
    max_files_changed: int = 20,
    min_python_files: int = 1,
    min_diff_chars: int = 50,
    skip_patterns: Optional[list[str]] = None,
    bot_suffixes: Optional[list[str]] = None,
):
    """Load raw PRs, run GitHub API enrichment, write filtered results."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_prs = read_jsonl(input_file)

    logger.info(f"Loaded {len(raw_prs)} raw PR records")
    logger.info(
        "Filtering settings: max_records=%s max_workers=%s max_diff_chars=%s "
        "max_files_changed=%s min_python_files=%s min_diff_chars=%s",
        max_records,
        max_workers,
        max_diff_chars,
        max_files_changed,
        min_python_files,
        min_diff_chars,
    )

    session = get_github_session()
    total_written = 0
    existing_ids = load_jsonl_id_set(output_path)
    total_written = len(existing_ids)

    out_f = open(output_path, "a")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_pr,
                    session,
                    pr,
                    max_diff_chars,
                    max_files_changed,
                    min_python_files,
                    min_diff_chars,
                    skip_patterns,
                    bot_suffixes,
                ): pr
                for pr in raw_prs
                if f"{pr['repo'].replace('/', '__')}-{pr['pr_number']}" not in existing_ids
            }

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Filtering PRs")
            for future in pbar:
                if max_records and total_written >= max_records:
                    for f in futures:
                        f.cancel()
                    break

                result = future.result()
                if result is not None:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    total_written += 1

                pbar.set_postfix({"kept": total_written})
    finally:
        out_f.close()

    logger.info(f"Filtered: {total_written} records written to {output_file}")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Filter PRs with GitHub API enrichment",
        epilog="""
Examples:
  # Basic usage (uses GITHUB_TOKEN env var)
  python -m data.filter_prs \\
    --input_file data/raw/raw_prs.jsonl \\
    --max_records 100

  # With token embedded (for cluster)
  python -m data.filter_prs \\
    --input_file data/raw/raw_prs.jsonl \\
    --token <your-github-token> \\
    --max_records 500 \\
    --max_workers 32
        """
    )
    parser.add_argument("--input_file", default="data/raw/raw_prs.jsonl")
    parser.add_argument("--output_file", default="data/raw/filtered_prs.jsonl")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_diff_chars", type=int, default=500_000)
    parser.add_argument("--max_files_changed", type=int, default=20)
    parser.add_argument("--min_python_files", type=int, default=1)
    parser.add_argument("--min_diff_chars", type=int, default=50)
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub token (overrides GITHUB_TOKEN env var). "
             "Get one from: https://github.com/settings/tokens"
    )
    parser.add_argument(
        "--bot_suffixes",
        nargs="+",
        default=[],
        help="Bot author name suffixes to skip (e.g. 'bot' 'app'). "
             "Authors ending with these suffixes will be filtered out."
    )
    args = parser.parse_args()

    # Set token in environment if provided
    if args.token:
        os.environ["GITHUB_TOKEN"] = args.token
        logger.info("Using GitHub token from --token argument")
    elif os.environ.get("GITHUB_TOKEN"):
        logger.info("Using GitHub token from GITHUB_TOKEN environment variable")
    else:
        logger.warning("⚠️  No GitHub token found! Filtering will be ~100x slower.")
        logger.warning("   To speed up: add --token ghp_xxx or set GITHUB_TOKEN env var")

    filter_prs(
        input_file=args.input_file,
        output_file=args.output_file,
        max_records=args.max_records,
        max_workers=args.max_workers,
        max_diff_chars=args.max_diff_chars,
        max_files_changed=args.max_files_changed,
        min_python_files=args.min_python_files,
        min_diff_chars=args.min_diff_chars,
        bot_suffixes=args.bot_suffixes,
    )


if __name__ == "__main__":
    main()
