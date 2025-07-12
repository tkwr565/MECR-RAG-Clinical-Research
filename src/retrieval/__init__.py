"""
Retrieval module for the Emergency Medicine Triage RAG System.

This module handles all retrieval operations including guideline selection,
specialty prediction, and past case similarity search with metadata filtering.

Components:
- guideline_retriever: Selection and retrieval of relevant triage guideline sections
- specialty_predictor: Prediction of appropriate medical specialties for cases
- case_retriever: Similarity search and retrieval of past cases with metadata filtering
- utils: Utility functions for metadata filtering and retrieval operations

Key Features:
- Optimized guideline selection (max 2 sections) based on clinical presentation
- Multi-specialty prediction with primary/secondary specialty support
- Vector similarity search with demographic and specialty filtering
- Metadata-based filtering using age groups and attending specialties
- Configurable similarity thresholds for high-quality case retrieval
"""

from .guideline_retriever import (
    create_guideline_selection_prompt,
    format_guideline_metadata,
    select_guideline_sections,
    retrieve_guideline_content,
    optimized_guideline_retrieval_decision,
    retrieved_guideline_section_content,
    validate_guideline_selection,
    format_guideline_content_for_prompt,
    get_guideline_selection_statistics,
    search_guidelines_by_keywords,
)

from .specialty_predictor import (
    create_specialty_prediction_prompt,
    predict_attending_specialty,
    attending_specialty_decision,
    validate_specialty_prediction,
    get_specialty_confidence_score,
    format_specialty_prediction_output,
    get_specialty_statistics,
)

from .case_retriever import (
    retrieve_past_cases_optimized,
    retrieve_past_case_optimized,
    retrieve_case_content_from_json,
    retrieve_simplified_case_content,
    format_past_cases_context,
    validate_retrieved_cases,
    get_case_retrieval_statistics,
)

from .utils import (
    filter_documents_by_metadata,
    extract_age_group_from_case,
    create_metadata_filters,
    get_filtered_document_ids,
    load_case_content_from_json,
    simplify_case_content,
    calculate_similarity_score,
    format_retrieved_case_result,
    validate_metadata_filters,
    get_metadata_statistics,
)

__all__ = [
    # Guideline retrieval
    "create_guideline_selection_prompt",
    "format_guideline_metadata",
    "select_guideline_sections",
    "retrieve_guideline_content",
    "optimized_guideline_retrieval_decision",
    "retrieved_guideline_section_content",
    "validate_guideline_selection",
    "format_guideline_content_for_prompt",
    "get_guideline_selection_statistics",
    "search_guidelines_by_keywords",
    # Specialty prediction
    "create_specialty_prediction_prompt",
    "predict_attending_specialty",
    "attending_specialty_decision",
    "validate_specialty_prediction",
    "get_specialty_confidence_score",
    "format_specialty_prediction_output",
    "get_specialty_statistics",
    # Case retrieval
    "retrieve_past_cases_optimized",
    "retrieve_past_case_optimized",
    "retrieve_case_content_from_json",
    "retrieve_simplified_case_content",
    "format_past_cases_context",
    "validate_retrieved_cases",
    "get_case_retrieval_statistics",
    # Utilities
    "filter_documents_by_metadata",
    "extract_age_group_from_case",
    "create_metadata_filters",
    "get_filtered_document_ids",
    "load_case_content_from_json",
    "simplify_case_content",
    "calculate_similarity_score",
    "format_retrieved_case_result",
    "validate_metadata_filters",
    "get_metadata_statistics",
]
