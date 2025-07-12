"""
Utilities module for the Emergency Medicine Triage RAG System.

This module provides utility functions and parsers used throughout the system
for data validation, output parsing, file operations, and common tasks.

Components:
- parsers: Custom output parsers for structured LLM responses
- helpers: General utility functions for file operations and data manipulation

Parser Features:
- SpecialtyPrediction: JSON parser for specialty prediction with validation
- TriageCategoryParser: Parser for triage category responses (1-5)
- GuidelineSectionParser: Parser for guideline section selections
- Safe JSON parsing with fallback handling

Helper Features:
- File operations (safe JSON load/save, directory management)
- Text processing (cleaning, truncation, number extraction)
- Data validation and structure checking
- Performance measurement and backup utilities
"""

from .parsers import (
    SpecialtyPrediction,
    TriageCategoryParser,
    GuidelineSectionParser,
    parse_json_safely,
    extract_category_from_text,
    validate_parser_output,
)

from .helpers import (
    ensure_directory_exists,
    get_file_hash,
    safe_json_load,
    safe_json_save,
    format_file_size,
    get_file_info,
    clean_text,
    truncate_text,
    flatten_dict,
    unflatten_dict,
    measure_execution_time,
    validate_json_structure,
    deep_merge_dicts,
    extract_numbers_from_text,
    create_backup_filename,
    get_nested_value,
    set_nested_value,
    calculate_percentage,
)

__all__ = [
    # Parsers
    "SpecialtyPrediction",
    "TriageCategoryParser",
    "GuidelineSectionParser",
    "parse_json_safely",
    "extract_category_from_text",
    "validate_parser_output",
    # Helpers
    "ensure_directory_exists",
    "get_file_hash",
    "safe_json_load",
    "safe_json_save",
    "format_file_size",
    "get_file_info",
    "clean_text",
    "truncate_text",
    "flatten_dict",
    "unflatten_dict",
    "measure_execution_time",
    "validate_json_structure",
    "deep_merge_dicts",
    "extract_numbers_from_text",
    "create_backup_filename",
    "get_nested_value",
    "set_nested_value",
    "calculate_percentage",
]
