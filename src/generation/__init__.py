"""
Generation module for the Emergency Medicine Triage RAG System.

This module handles the final triage assessment generation using a structured 3-step
reasoning approach that integrates clinical risk assessment, guideline-based evaluation,
and real-world factors analysis from past cases.

Components:
- triage_assessor: Comprehensive triage category prediction and reasoning

Assessment Framework:
- Step 1: Clinical Risk Assessment based on triage category definitions
- Step 2: Guideline-Based Assessment using specific medical condition guidelines
- Step 3: Real-World Factors Analysis incorporating past case patterns

Output Features:
- Structured 3-step reasoning process
- Final triage category (1-5) with confidence level
- Detailed rationale explaining decision factors
- Integration of guidelines and past case insights
- Extraction and validation of assessment components
"""

from .triage_assessor import (
    create_triage_assessment_prompt,
    prepare_case_summary_for_assessment,
    generate_triage_assessment,
    simplified_triage_prediction,
    extract_final_category_from_assessment,
    validate_triage_assessment,
    get_assessment_statistics,
)

__all__ = [
    "create_triage_assessment_prompt",
    "prepare_case_summary_for_assessment",
    "generate_triage_assessment",
    "simplified_triage_prediction",
    "extract_final_category_from_assessment",
    "validate_triage_assessment",
    "get_assessment_statistics",
]
