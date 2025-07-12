"""
Preprocessing module for the Emergency Medicine Triage RAG System.

This module handles the initial preprocessing of case data into concise clinical summaries
that are optimized for vector embedding retrieval and similarity search.

Components:
- summarizer: Case summarization functions for creating clinical summaries

Key Features:
- Extracts relevant demographics, vitals, and clinical presentation
- Applies medical terminology guidelines and abbreviation expansion
- Focuses on abnormal vital signs based on strict clinical criteria
- Generates 2-3 sentence summaries optimized for retrieval
- Validates summary quality and completeness
"""

from .summarizer import (
    extract_sections_for_summary,
    create_summary_prompt,
    process_case_summary,
    generate_summary_standalone,
    validate_summary,
)

__all__ = [
    "extract_sections_for_summary",
    "create_summary_prompt",
    "process_case_summary",
    "generate_summary_standalone",
    "validate_summary",
]
