"""
io_utils.py
===========
File I/O utilities for configuration, data, and results.

Handles:
  - YAML config loading
  - JSONL data file reading/writing
  - JSON file operations
  - Directory creation
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


def read_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if yaml is None:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise


def write_yaml(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Write a configuration dictionary to YAML file."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if yaml is None:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml")

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {output_path}")


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read a JSONL (JSON Lines) file.

    Each line is a JSON object. Returns list of dictionaries.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries, one per line

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If a line is invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"JSONL file not found: {file_path}, returning empty list")
        return []

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON at line {line_num}: {e}")
                raise

    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def write_jsonl(records: List[Dict[str, Any]], file_path: Union[str, Path], mode: str = "w") -> None:
    """
    Write records to a JSONL file.

    Args:
        records: List of dictionaries to write
        file_path: Path to output JSONL file
        mode: 'w' for write (overwrite), 'a' for append
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, mode, encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Wrote {len(records)} records to {file_path}")


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {file_path}")
    return data


def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info(f"Saved JSON to {file_path}")


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory

    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_text(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """Read a text file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_text(text: str, file_path: Union[str, Path], encoding: str = "utf-8") -> None:
    """Write text to a file."""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(text)

    logger.info(f"Wrote text to {file_path}")


def count_files(directory: Union[str, Path], pattern: str = "*") -> int:
    """Count files in a directory matching a pattern."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes."""
    file_path = Path(file_path)
    return file_path.stat().st_size / (1024 * 1024)


def append_jsonl(records: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Append records to a JSONL file (append mode).

    Args:
        records: List of dictionaries to append
        file_path: Path to JSONL file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "a", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Appended {len(records)} records to {file_path}")


def load_jsonl_id_set(file_path: Union[str, Path], id_field: str = "instance_id") -> set:
    """
    Load a set of IDs from a JSONL file (for resume/deduplication).

    Args:
        file_path: Path to JSONL file
        id_field: Field name containing the ID

    Returns:
        Set of IDs from the file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return set()

    ids = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get(id_field)
                    if record_id:
                        ids.add(str(record_id))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning(f"Failed to load ID set from {file_path}: {e}")

    return ids


def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
    """
    Ensure parent directory of a file exists.

    Args:
        file_path: Path to file

    Returns:
        Parent directory path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path.parent


def append_jsonl(records: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Append records to a JSONL file (append mode).

    Args:
        records: List of dictionaries to append
        file_path: Path to JSONL file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "a", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Appended {len(records)} records to {file_path}")


def load_jsonl_id_set(file_path: Union[str, Path], id_field: str = "instance_id") -> set:
    """
    Load a set of IDs from a JSONL file (for resume/deduplication).

    Args:
        file_path: Path to JSONL file
        id_field: Field name containing the ID

    Returns:
        Set of IDs from the file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return set()

    ids = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get(id_field)
                    if record_id:
                        ids.add(str(record_id))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning(f"Failed to load ID set from {file_path}: {e}")

    return ids


def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
    """
    Ensure parent directory of a file exists.

    Args:
        file_path: Path to file

    Returns:
        Parent directory path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path.parent


def append_jsonl(records: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Append records to a JSONL file (append mode).

    Args:
        records: List of dictionaries to append
        file_path: Path to JSONL file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "a", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Appended {len(records)} records to {file_path}")


def load_jsonl_id_set(file_path: Union[str, Path], id_field: str = "instance_id") -> set:
    """
    Load a set of IDs from a JSONL file (for resume/deduplication).

    Args:
        file_path: Path to JSONL file
        id_field: Field name containing the ID

    Returns:
        Set of IDs from the file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return set()

    ids = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get(id_field)
                    if record_id:
                        ids.add(str(record_id))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning(f"Failed to load ID set from {file_path}: {e}")

    return ids


def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
    """
    Ensure parent directory of a file exists.

    Args:
        file_path: Path to file

    Returns:
        Parent directory path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path.parent
