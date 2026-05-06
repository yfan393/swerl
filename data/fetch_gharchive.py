"""
fetch_gharchive.py
==================
Download and parse GHArchive event dumps (hourly JSON.gz files) to extract
merged pull request events from public GitHub repositories.

GHArchive stores one file per hour at:
  https://data.gharchive.org/{YYYY-MM-DD}-{H}.json.gz

We filter for PullRequestEvent where:
  - action == "closed"
  - pull_request.merged == True
  - at least one Python file changed (checked later via GitHub API)

Usage:
    python -m data.fetch_gharchive \
        --start_date 2023-01-01 \
        --end_date 2023-01-07 \
        --output_file data/raw/raw_prs.jsonl \
        --max_prs 50000
"""

import argparse
import gzip
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GHARCHIVE_BASE = "https://data.gharchive.org"

# Regex patterns that suggest a PR is related to a bug fix
BUG_FIX_PATTERNS = re.compile(
    r"\b(fix|bug|patch|issue|error|defect|close[sd]?|resolve[sd]?|repair)\b",
    re.IGNORECASE,
)

# Patterns indicating a PR closes/fixes a numbered issue
CLOSES_ISSUE_PATTERN = re.compile(
    r"(close[sd]?|fix(e[sd])?|resolve[sd]?)\s*#\d+",
    re.IGNORECASE,
)


def iter_hourly_urls(start_date: str, end_date: str) -> Generator[str, None, None]:
    """Yield GHArchive download URLs for every hour in [start_date, end_date]."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    while current < end:
        for hour in range(24):
            date_str = current.strftime("%Y-%m-%d")
            yield f"{GHARCHIVE_BASE}/{date_str}-{hour}.json.gz"
        current += timedelta(days=1)


def download_and_parse(url: str, max_retries: int = 3) -> list[dict]:
    """
    Download a single GHArchive hourly dump and return all merged PR events.

    Returns a list of dicts with keys:
        repo, pr_number, pr_title, pr_body, merged_at,
        base_sha, head_sha, author
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 404:
                return []  # Hour file may not exist
            resp.raise_for_status()

            results = []
            buf = io.BytesIO(resp.content)
            with gzip.open(buf, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") != "PullRequestEvent":
                        continue

                    payload = event.get("payload", {})
                    if payload.get("action") != "closed":
                        continue

                    pr = payload.get("pull_request", {})
                    if not pr.get("merged"):
                        continue

                    repo = event.get("repo", {}).get("name", "")
                    if not repo:
                        continue

                    pr_body = pr.get("body") or ""
                    pr_title = pr.get("title") or ""

                    # Quick heuristic: must look like a bug fix
                    combined_text = pr_title + " " + pr_body
                    if not BUG_FIX_PATTERNS.search(combined_text):
                        continue

                    # Must reference an issue number
                    if not CLOSES_ISSUE_PATTERN.search(combined_text):
                        continue

                    author = (pr.get("user") or {}).get("login", "")
                    results.append(
                        {
                            "repo": repo,
                            "pr_number": pr.get("number"),
                            "pr_title": pr_title,
                            "pr_body": pr_body[:4000],  # Truncate huge bodies
                            "merged_at": pr.get("merged_at", ""),
                            "base_sha": (pr.get("base") or {}).get("sha", ""),
                            "head_sha": (pr.get("head") or {}).get("sha", ""),
                            "author": author,
                            "html_url": pr.get("html_url", ""),
                        }
                    )
            return results

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Retry {attempt+1}/{max_retries} for {url}: {e}. Waiting {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"Failed to fetch {url}: {e}")
                return []


def fetch_prs(
    start_date: str,
    end_date: str,
    output_file: str,
    max_prs: Optional[int] = None,
    max_workers: int = 4,
):
    """
    Main function: iterate all GHArchive hours in range, download in parallel,
    and write merged PR metadata to a JSONL file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    urls = list(iter_hourly_urls(start_date, end_date))
    logger.info(f"Total hourly files to process: {len(urls)}")

    total_written = 0
    seen_prs: set[str] = set()  # deduplicate by (repo, pr_number)

    with open(output_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_and_parse, url): url for url in urls}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="GHArchive hours")

            for future in pbar:
                if max_prs and total_written >= max_prs:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                prs = future.result()
                for pr in prs:
                    key = f"{pr['repo']}#{pr['pr_number']}"
                    if key in seen_prs:
                        continue
                    seen_prs.add(key)
                    out_f.write(json.dumps(pr) + "\n")
                    total_written += 1

                pbar.set_postfix({"prs": total_written})

    logger.info(f"Wrote {total_written} merged PR records to {output_file}")
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Fetch merged PRs from GHArchive")
    parser.add_argument("--start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output_file", default="data/raw/raw_prs.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--max_prs", type=int, default=None, help="Stop after collecting this many PRs"
    )
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel downloads")
    args = parser.parse_args()

    fetch_prs(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output_file,
        max_prs=args.max_prs,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
