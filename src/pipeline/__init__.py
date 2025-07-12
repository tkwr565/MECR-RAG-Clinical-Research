"""
Pipeline module for the Emergency Medicine Triage RAG System.

This module orchestrates the complete triage assessment workflow using LangGraph
for state management and provides high-level processing functions for single cases
and batch operations.

Components:
- graph_builder: LangGraph construction and node orchestration
- processor: High-level processing functions for cases and batch operations

Pipeline Flow:
1. Case Preprocessing (Summarization)
2. Guideline Retrieval (Max 2 sections)
3. Specialty Prediction
4. Past Case Retrieval (Metadata + Similarity filtering)
5. Final Triage Assessment (3-step reasoning)

Processing Options:
- Single case processing from file or data
- Batch directory processing with progress tracking
- Optimized pipeline with configurable parameters
- Comprehensive error handling and validation
"""

from .graph_builder import (
    create_node_functions,
    build_optimized_triage_graph,
    create_initial_state,
    validate_graph_inputs,
    get_graph_info,
    execute_graph_with_validation,
)

from .processor import (
    process_case_file_optimized,
    process_case_data_optimized,
    batch_process_directory_optimized,
    create_output_summary,
    create_enhanced_output_summary,  # Added new function
    validate_pipeline_setup,
    get_pipeline_info,
)

__all__ = [
    # Graph building and management
    "create_node_functions",
    "build_optimized_triage_graph",
    "create_initial_state",
    "validate_graph_inputs",
    "get_graph_info",
    "execute_graph_with_validation",
    # High-level processing
    "process_case_file_optimized",
    "process_case_data_optimized",
    "batch_process_directory_optimized",
    "create_output_summary",
    "create_enhanced_output_summary",  # Added new function
    "validate_pipeline_setup",
    "get_pipeline_info",
]
