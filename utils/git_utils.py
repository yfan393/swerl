"""
git_utils.py
============
Git and code utilities for SWE-RL.

Handles:
  - Patch application and validation
  - Code syntax checking
  - Linting and code quality
  - Diff parsing
"""

import ast
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_valid_python_syntax(code: str, filename: str = "<code>") -> bool:
    """
    Check if code has valid Python syntax.

    Args:
        code: Python code to check
        filename: Optional filename for error messages

    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.debug(f"Syntax error in {filename}: {e}")
        return False


def lint_code(code: str, filename: str = "temp.py") -> List[str]:
    """
    Run flake8 linting on code.

    Args:
        code: Python code to lint
        filename: Optional filename for linting

    Returns:
        List of linting errors (empty if no errors)
    """
    try:
        import flake8.api.get_style_guide
    except ImportError:
        logger.warning("flake8 not installed, skipping linting")
        return []

    try:
        # Write to temp file for linting
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Run flake8
        style_guide = flake8.api.get_style_guide(quiet=True)
        report = style_guide.check_files([temp_path])

        # Extract error messages
        errors = []
        if report._deferred_print:
            errors = report._deferred_print

        # Clean up
        Path(temp_path).unlink()

        return errors

    except Exception as e:
        logger.warning(f"Flake8 check failed: {e}")
        return []


def check_code_differ_by_just_empty_lines(code1: str, code2: str) -> bool:
    """
    Check if two code blocks differ only in whitespace/empty lines.

    Args:
        code1: First code block
        code2: Second code block

    Returns:
        True if they differ only by empty lines, False otherwise
    """
    lines1 = [l.rstrip() for l in code1.split("\n") if l.strip()]
    lines2 = [l.rstrip() for l in code2.split("\n") if l.strip()]
    return lines1 == lines2


def apply_patch(
    original_code: Dict[str, str], patch_dict: Dict[str, List[Tuple[str, str]]]
) -> Tuple[Dict[str, str], bool]:
    """
    Apply SEARCH/REPLACE patches to code.

    Args:
        original_code: Dict of {filepath: code_content}
        patch_dict: Dict of {filepath: [(search, replace), ...]}

    Returns:
        Tuple of (patched_code, success)
        - patched_code: Dict of {filepath: patched_content}
        - success: Whether all patches applied cleanly
    """
    patched = {}
    success = True

    for filepath, pairs in patch_dict.items():
        if filepath not in original_code:
            logger.warning(f"File not in original code: {filepath}")
            success = False
            continue

        content = original_code[filepath]

        for search, replace in pairs:
            # Normalize whitespace
            if search not in content:
                logger.warning(f"Search pattern not found in {filepath}")
                logger.debug(f"Search: {repr(search[:50])}")
                logger.debug(f"Content sample: {repr(content[:100])}")
                success = False
                continue

            content = content.replace(search, replace, 1)

        patched[filepath] = content

    return patched, success


def parse_search_replace_blocks(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse SEARCH/REPLACE blocks from model output.

    Format:
    ```
    ### filepath.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE
    ```

    Args:
        text: Model output containing SEARCH/REPLACE blocks

    Returns:
        Dict of {filepath: [(search, replace), ...]}
    """
    # Regex pattern for SEARCH/REPLACE blocks
    pattern = (
        r"```[^\n]*\n"
        r"###\s*([^\n]+)\n"
        r"<<<<<<< SEARCH\n"
        r"([\s\S]*?)\n"
        r"=======\n"
        r"([\s\S]*?)\n"
        r">>>>>>> REPLACE\n"
        r"```"
    )

    results = {}
    for match in re.finditer(pattern, text):
        filepath = match.group(1).strip()
        search = match.group(2)
        replace = match.group(3)

        if filepath not in results:
            results[filepath] = []
        results[filepath].append((search, replace))

    return results


def extract_thought_solution(output: str) -> Tuple[str, str]:
    """
    Extract <think> and <solution> blocks from model output.

    Args:
        output: Model output text

    Returns:
        Tuple of (thought, solution)

    Raises:
        ValueError: If blocks are not found or malformed
    """
    # Check for required tags
    for tag in ["<think>", "</think>", "<solution>", "</solution>"]:
        if output.count(tag) != 1:
            raise ValueError(f"Expected exactly one {tag} tag, found {output.count(tag)}")

    # Extract blocks
    try:
        thought_start = output.index("<think>") + len("<think>")
        thought_end = output.index("</think>")
        thought = output[thought_start:thought_end].strip()

        solution_start = output.index("<solution>") + len("<solution>")
        solution_end = output.index("</solution>")
        solution = output[solution_start:solution_end].strip()
    except ValueError as e:
        raise ValueError(f"Failed to extract thought/solution blocks: {e}")

    if not thought:
        raise ValueError("Thought block is empty")
    if not solution:
        raise ValueError("Solution block is empty")

    return thought, solution


def create_unified_diff(original: str, modified: str, filepath: str) -> str:
    """
    Create a unified diff between two code versions.

    Args:
        original: Original code
        modified: Modified code
        filepath: Path for diff header

    Returns:
        Unified diff string
    """
    import difflib

    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm="",
    )

    return "\n".join(diff)


def get_changed_files(diff_text: str) -> List[str]:
    """Extract list of changed files from a unified diff."""
    pattern = r"^---\s+a/(.+?)$"
    files = re.findall(pattern, diff_text, re.MULTILINE)
    return files


def normalize_patch(search: str, replace: str) -> Tuple[str, str]:
    """
    Normalize patch text (trim extra whitespace).

    Args:
        search: Search text
        replace: Replace text

    Returns:
        Tuple of normalized (search, replace)
    """
    # Strip common leading whitespace
    search_lines = search.splitlines()
    replace_lines = replace.splitlines()

    # Find minimum indentation
    def min_indent(lines):
        min_indent_val = float("inf")
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent_val = min(min_indent_val, indent)
        return min_indent_val if min_indent_val != float("inf") else 0

    search_indent = min_indent(search_lines)
    replace_indent = min_indent(replace_lines)

    # Remove common indentation
    search_normalized = "\n".join(
        line[search_indent:] if len(line) > search_indent else line
        for line in search_lines
    ).strip()
    replace_normalized = "\n".join(
        line[replace_indent:] if len(line) > replace_indent else line
        for line in replace_lines
    ).strip()

    return search_normalized, replace_normalized


def has_new_syntax_errors(original: str, modified: str) -> bool:
    """
    Check if modified code has syntax errors that didn't exist in original.

    Args:
        original: Original code
        modified: Modified code

    Returns:
        True if new syntax errors were introduced
    """
    original_valid = is_valid_python_syntax(original)
    modified_valid = is_valid_python_syntax(modified)

    # If original was invalid, we don't care about new errors
    if not original_valid:
        return False

    # If modified is invalid but original was valid, that's a new error
    return not modified_valid
