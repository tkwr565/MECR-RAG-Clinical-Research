"""
Emergency Medicine Triage RAG System

A comprehensive Retrieval-Augmented Generation system for emergency medicine triage
decision support that processes new patient cases by analyzing case descriptions,
retrieving relevant triage guidelines and similar past cases, and generating
comprehensive triage assessments.

Main Components:
- Case preprocessing and summarization
- Guideline retrieval and selection
- Specialty prediction
- Past case similarity search
- Final triage assessment generation

Usage:
    from src.pipeline.processor import process_case_file_optimized
    from src.models.llm_factory import create_llm
    from src.data.loaders import load_vector_database

    # Initialize components
    llm = create_llm("gpt4o")
    vectordb = load_vector_database("gpt-4o")

    # Process a case
    result = process_case_file_optimized(
        file_path="path/to/case.json",
        llm=llm,
        summary_vectordb=vectordb,
        case_json_dir="db/past_case/db_gpt-4o_3000case/json_store"
    )
"""

__version__ = "1.0.0"
__author__ = "Emergency Medicine Triage RAG Team"

# Import key classes and functions for easy access
from .config.settings import settings
from .config.specialty_mapping import AGENT_SPECIALTY_LIST
from .data.state import GraphState
from .models.llm_factory import create_llm, get_available_models
from .models.embeddings import get_default_embeddings
from .pipeline.processor import (
    process_case_file_optimized,
    process_case_data_optimized,
    batch_process_directory_optimized,
)
from .pipeline.graph_builder import build_optimized_triage_graph

__all__ = [
    # Core functionality
    "process_case_file_optimized",
    "process_case_data_optimized",
    "batch_process_directory_optimized",
    "build_optimized_triage_graph",
    # Models and embeddings
    "create_llm",
    "get_available_models",
    "get_default_embeddings",
    # Configuration and state
    "settings",
    "AGENT_SPECIALTY_LIST",
    "GraphState",
    # Version info
    "__version__",
    "__author__",
]
