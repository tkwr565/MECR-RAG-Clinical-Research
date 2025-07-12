"""
Configuration module for the Emergency Medicine Triage RAG System.

This module handles all configuration settings, environment variables,
and specialty mappings used throughout the system.

Components:
- settings: Global configuration settings and environment variables
- specialty_mapping: Medical specialty mappings and utility functions
"""

from .settings import settings, MODEL_NAMES, get_model_db_name
from .specialty_mapping import (
    AGENT_SPECIALTY_LIST,
    SPECIALTY_MAPPING,
    REVERSE_SPECIALTY_MAPPING,
    map_specialty_to_dataset_values,
    get_standardized_specialty,
    validate_specialty,
    get_all_dataset_values,
    get_specialty_info,
)

__all__ = [
    # Settings
    "settings",
    "MODEL_NAMES",
    "get_model_db_name",
    # Specialty mappings
    "AGENT_SPECIALTY_LIST",
    "SPECIALTY_MAPPING",
    "REVERSE_SPECIALTY_MAPPING",
    "map_specialty_to_dataset_values",
    "get_standardized_specialty",
    "validate_specialty",
    "get_all_dataset_values",
    "get_specialty_info",
]
