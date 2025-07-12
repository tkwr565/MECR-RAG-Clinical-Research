"""
Data module for the Emergency Medicine Triage RAG System.

This module handles data loading, state management, and data structure definitions
used throughout the triage assessment pipeline.

Components:
- loaders: Functions for loading case data, guidelines, and vector databases
- state: GraphState class definition for pipeline state management

Data Sources:
- Case JSON files: Individual patient case data
- Triage guidelines: Medical guideline sections with summaries
- Vector databases: Past case embeddings for similarity search
"""

from .loaders import (
    load_case_from_json,
    load_triage_guidelines,
    extract_guideline_metadata,
    load_vector_database,
    validate_vector_database,
    get_case_json_directory,
    load_case_json_from_id,
    check_data_availability,
    get_available_models,
)

from .state import GraphState

__all__ = [
    # Data loaders
    "load_case_from_json",
    "load_triage_guidelines",
    "extract_guideline_metadata",
    "load_vector_database",
    "validate_vector_database",
    "get_case_json_directory",
    "load_case_json_from_id",
    "check_data_availability",
    "get_available_models",
    # State management
    "GraphState",
]
