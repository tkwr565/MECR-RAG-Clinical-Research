"""
General utility functions for the Emergency Medicine Triage RAG System.

This module contains helper functions for file operations, data validation,
formatting, and other common operations used across the system.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file for change detection.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return ""


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file with error handling.

    Args:
        file_path: Path to JSON file
        default: Default value if loading fails

    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Safely save data to JSON file with error handling.

    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists - handle case where dirname might be empty
        output_dir = os.path.dirname(file_path)
        if output_dir:  # Only create directory if dirname is not empty
            ensure_directory_exists(output_dir)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    try:
        stat = os.stat(file_path)
        return {
            "exists": True,
            "size": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "extension": os.path.splitext(file_path)[1].lower(),
            "hash": get_file_hash(file_path),
        }
    except FileNotFoundError:
        return {"exists": False, "error": "File not found"}
    except Exception as e:
        return {"exists": False, "error": str(e)}


def clean_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """
    Clean text by removing or normalizing whitespace and special characters.

    Args:
        text: Text to clean
        remove_extra_whitespace: Whether to collapse multiple whitespace

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)

    # Remove null characters and other problematic characters
    text = text.replace("\x00", "").replace("\r", "")

    if remove_extra_whitespace:
        # Collapse multiple whitespace characters
        import re

        text = re.sub(r"\s+", " ", text)

    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Unflatten a flattened dictionary.

    Args:
        d: Flattened dictionary
        sep: Separator used in keys

    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def measure_execution_time(func):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that prints execution time
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper


def validate_json_structure(
    data: Dict[str, Any], required_fields: List[str]
) -> Dict[str, Any]:
    """
    Validate JSON data structure against required fields.

    Args:
        data: JSON data to validate
        required_fields: List of required field names

    Returns:
        Validation result dictionary
    """
    validation = {"valid": True, "missing_fields": [], "extra_info": {}}

    if not isinstance(data, dict):
        validation["valid"] = False
        validation["error"] = "Data is not a dictionary"
        return validation

    # Check required fields
    for field in required_fields:
        if field not in data:
            validation["missing_fields"].append(field)

    if validation["missing_fields"]:
        validation["valid"] = False

    # Additional info
    validation["extra_info"] = {
        "total_fields": len(data),
        "required_fields_count": len(required_fields),
        "missing_count": len(validation["missing_fields"]),
    }

    return validation


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract all numbers from text string.

    Args:
        text: Text to extract numbers from

    Returns:
        List of numbers found
    """
    import re

    # Pattern to match integers and floats (including negative)
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)

    numbers = []
    for match in matches:
        try:
            if "." in match:
                numbers.append(float(match))
            else:
                numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def create_backup_filename(original_path: str, timestamp: bool = True) -> str:
    """
    Create a backup filename for a given file path.

    Args:
        original_path: Original file path
        timestamp: Whether to include timestamp

    Returns:
        Backup filename
    """
    base, ext = os.path.splitext(original_path)

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_backup_{timestamp_str}{ext}"
    else:
        return f"{base}_backup{ext}"


def get_nested_value(
    data: Dict[str, Any], key_path: str, default: Any = None, sep: str = "."
) -> Any:
    """
    Get value from nested dictionary using dot notation.

    Args:
        data: Dictionary to search
        key_path: Dot-separated key path
        default: Default value if key not found
        sep: Separator character

    Returns:
        Value at key path or default
    """
    keys = key_path.split(sep)
    current = data

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_nested_value(
    data: Dict[str, Any], key_path: str, value: Any, sep: str = "."
) -> None:
    """
    Set value in nested dictionary using dot notation.

    Args:
        data: Dictionary to modify
        key_path: Dot-separated key path
        value: Value to set
        sep: Separator character
    """
    keys = key_path.split(sep)
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def calculate_percentage(part: Union[int, float], total: Union[int, float]) -> float:
    """
    Calculate percentage with division by zero protection.

    Args:
        part: Part value
        total: Total value

    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100
