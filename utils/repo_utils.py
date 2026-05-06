"""
repo_utils.py
=============
Git repository operations for cloning, checking out commits, and reading files.

Handles:
  - Repository cloning and caching
  - Commit availability checking
  - File listing at specific commits
  - File reading at specific commits
  - Temporary directory management
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache of cloned repositories to avoid repeated clones
_REPO_CACHE: Dict[str, Path] = {}


def get_cache_dir() -> Path:
    """Get or create a cache directory for cloned repos."""
    cache_dir = Path(tempfile.gettempdir()) / "swerl_repo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clone_repo(repo_url: str, repo_name: str = None, force: bool = False) -> Path:
    """
    Clone a repository and cache it locally.

    Args:
        repo_url: URL of the repository to clone
        repo_name: Optional custom name for the cached directory
        force: If True, reclone even if cached

    Returns:
        Path to the cloned repository
    """
    if repo_name is None:
        # Extract repo name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

    cache_dir = get_cache_dir()
    repo_path = cache_dir / repo_name

    # Check cache
    if not force and repo_path.exists():
        logger.debug(f"Using cached repo: {repo_path}")
        return repo_path

    # Remove if force=True and exists
    if force and repo_path.exists():
        shutil.rmtree(repo_path)

    try:
        logger.info(f"Cloning {repo_url} to {repo_path}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        logger.info(f"Successfully cloned to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone {repo_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error cloning {repo_url}: {e}")
        raise


def ensure_commit_available(
    repo_path: Path, commit: str, remote: str = "origin"
) -> bool:
    """
    Ensure a specific commit is available in the repository.
    Fetches from remote if necessary.

    Args:
        repo_path: Path to the repository
        commit: Commit hash to check/fetch
        remote: Remote name (default: "origin")

    Returns:
        True if commit is available, False otherwise
    """
    try:
        # Try to verify the commit exists
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=str(repo_path),
            capture_output=True,
            timeout=30,
        )

        if result.returncode == 0:
            return True

        # Commit not found, try fetching
        logger.info(f"Commit {commit} not found, attempting fetch from {remote}")
        subprocess.run(
            ["git", "fetch", remote, commit],
            cwd=str(repo_path),
            capture_output=True,
            timeout=60,
        )

        # Try again after fetch
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=str(repo_path),
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout checking commit {commit}")
        return False
    except Exception as e:
        logger.error(f"Error checking commit {commit}: {e}")
        return False


def list_files_at_commit(
    repo_path: Path, commit: str, pattern: str = "*.py"
) -> List[str]:
    """
    List all files at a specific commit.

    Args:
        repo_path: Path to the repository
        commit: Commit hash
        pattern: Optional file pattern filter (e.g., "*.py")

    Returns:
        List of file paths relative to repo root
    """
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to list files at {commit}")
            return []

        files = result.stdout.strip().split("\n")
        files = [f for f in files if f]  # Remove empty strings

        # Filter by pattern if needed
        if pattern and pattern != "*":
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]

        return files

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout listing files at {commit}")
        return []
    except Exception as e:
        logger.error(f"Error listing files at {commit}: {e}")
        return []


def read_file_at_commit(repo_path: Path, commit: str, file_path: str) -> Optional[str]:
    """
    Read a file at a specific commit.

    Args:
        repo_path: Path to the repository
        commit: Commit hash
        file_path: Path to the file relative to repo root

    Returns:
        File content as string, or None if not found/error
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{file_path}"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.debug(f"Failed to read {file_path} at {commit}")
            return None

        return result.stdout

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout reading {file_path} at {commit}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path} at {commit}: {e}")
        return None


def get_commit_metadata(repo_path: Path, commit: str) -> Optional[Dict[str, str]]:
    """
    Get metadata about a commit (author, date, message).

    Args:
        repo_path: Path to the repository
        commit: Commit hash

    Returns:
        Dict with 'author', 'date', 'message' keys, or None if error
    """
    try:
        # Get commit info in standardized format
        result = subprocess.run(
            ["git", "show", "-s", "--format=%an%n%aI%n%B", commit],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n", 2)
        if len(lines) < 3:
            return None

        return {
            "author": lines[0],
            "date": lines[1],
            "message": lines[2],
        }

    except Exception as e:
        logger.error(f"Error getting metadata for {commit}: {e}")
        return None


def checkout_commit(repo_path: Path, commit: str) -> bool:
    """
    Checkout a specific commit in the repository.

    Args:
        repo_path: Path to the repository
        commit: Commit hash

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout checking out {commit}")
        return False
    except Exception as e:
        logger.error(f"Error checking out {commit}: {e}")
        return False


def clear_repo_cache() -> None:
    """Clear all cached repositories."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"Cleared repo cache at {cache_dir}")


def ensure_repo_cloned(repo_slug: str, cache_dir: str = None) -> Path:
    """
    Ensure a repository (from GitHub) is cloned locally.

    Args:
        repo_slug: Repository slug in format "owner/name" or full HTTPS URL
        cache_dir: Optional custom cache directory

    Returns:
        Path to the cloned repository
    """
    # If repo_slug is a URL, use it directly; otherwise construct GitHub URL
    if repo_slug.startswith("http"):
        repo_url = repo_slug
        repo_name = repo_slug.rstrip("/").split("/")[-1].replace(".git", "")
    else:
        repo_url = f"https://github.com/{repo_slug}.git"
        repo_name = repo_slug.replace("/", "__")

    if cache_dir:
        # Use custom cache directory
        custom_cache = Path(cache_dir)
        custom_cache.mkdir(parents=True, exist_ok=True)
        repo_path = custom_cache / repo_name

        if repo_path.exists():
            logger.debug(f"Using cached repo: {repo_path}")
            return repo_path

        try:
            logger.info(f"Cloning {repo_url} to {repo_path}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                check=True,
                capture_output=True,
                timeout=300,
            )
            return repo_path
        except Exception as e:
            logger.error(f"Failed to clone {repo_url}: {e}")
            raise
    else:
        return clone_repo(repo_url, repo_name)
