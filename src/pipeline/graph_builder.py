"""
Graph builder module for the Emergency Medicine Triage RAG System.

This module constructs the LangGraph pipeline that orchestrates the complete
triage assessment workflow from case preprocessing to final assessment.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma

from ..data.state import GraphState
from ..data.loaders import load_triage_guidelines, extract_guideline_metadata
from ..config.settings import settings
from ..config.specialty_mapping import AGENT_SPECIALTY_LIST
from ..preprocessing.summarizer import process_case_summary
from ..retrieval.guideline_retriever import (
    optimized_guideline_retrieval_decision,
    retrieved_guideline_section_content,
)
from ..retrieval.specialty_predictor import attending_specialty_decision
from ..retrieval.case_retriever import (
    retrieve_past_case_optimized,
    retrieve_simplified_case_content,
)
from ..generation.triage_assessor import simplified_triage_prediction


def create_node_functions(
    llm,
    summary_vectordb: Chroma,
    case_json_dir: str,
    full_guidelines: List[Dict[str, str]],
    k_cases: int = None,
    metadata_threshold: float = None,
    similarity_threshold: float = None,
) -> Dict[str, callable]:
    """
    Create node functions with bound parameters for the graph.

    Args:
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing case JSON files
        full_guidelines: Complete guideline data
        k_cases: Number of past cases to retrieve
        metadata_threshold: Threshold for metadata filtering
        similarity_threshold: Minimum similarity score for past cases

    Returns:
        Dictionary of node functions ready for graph construction
    """
    # Use default values from settings if not provided
    if k_cases is None:
        k_cases = settings.K_CASES
    if metadata_threshold is None:
        metadata_threshold = settings.METADATA_THRESHOLD
    if similarity_threshold is None:
        similarity_threshold = settings.SIMILARITY_THRESHOLD

    # Create node functions with bound parameters
    def process_case_summary_node(state: GraphState) -> GraphState:
        return process_case_summary(state, llm)

    def guideline_retrieval_node(state: GraphState) -> GraphState:
        return optimized_guideline_retrieval_decision(
            state, llm, max_sections=settings.MAX_GUIDELINE_SECTIONS
        )

    def guideline_content_node(state: GraphState) -> GraphState:
        return retrieved_guideline_section_content(state, full_guidelines)

    def specialty_prediction_node(state: GraphState) -> GraphState:
        return attending_specialty_decision(state, llm)

    def case_retrieval_node(state: GraphState) -> GraphState:
        return retrieve_past_case_optimized(
            state=state,
            summary_vectordb=summary_vectordb,
            k=k_cases,
            metadata_threshold=metadata_threshold,
            similarity_threshold=similarity_threshold,
        )

    def case_content_node(state: GraphState) -> GraphState:
        return retrieve_simplified_case_content(state, case_json_dir)

    def triage_assessment_node(state: GraphState) -> GraphState:
        return simplified_triage_prediction(state, llm)

    return {
        "process_case_summary": process_case_summary_node,
        "optimized_guideline_retrieval": guideline_retrieval_node,
        "guideline_retrieve_content": guideline_content_node,
        "attending_specialty_decision": specialty_prediction_node,
        "retrieve_past_case_optimized": case_retrieval_node,
        "retrieve_simplified_case_content": case_content_node,
        "simplified_predict_category": triage_assessment_node,
    }


def build_optimized_triage_graph(
    llm,
    summary_vectordb: Chroma,
    case_json_dir: str,
    full_guidelines: List[Dict[str, str]] = None,
    k_cases: int = None,
    metadata_threshold: float = None,
    similarity_threshold: float = None,
) -> StateGraph:
    """
    Build an optimized triage assessment graph with simplified retrieval and reasoning.

    Args:
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing the case JSON files
        full_guidelines: Complete guideline data (optional, will load if not provided)
        k_cases: Number of past cases to retrieve
        metadata_threshold: Threshold for metadata filtering
        similarity_threshold: Minimum similarity score for past cases

    Returns:
        The compiled optimized graph
    """
    # Load guidelines if not provided
    if full_guidelines is None:
        full_guidelines = load_triage_guidelines()

    # Use default values from settings if not provided
    if k_cases is None:
        k_cases = settings.K_CASES
    if metadata_threshold is None:
        metadata_threshold = settings.METADATA_THRESHOLD
    if similarity_threshold is None:
        similarity_threshold = settings.SIMILARITY_THRESHOLD

    # Initialize the graph - use dict as state type like in your notebook
    graph = StateGraph(dict)

    # Wrap the optimized retrieval function with parameters - exactly like your notebook
    retrieve_past_case_optimized_with_params = (
        lambda state: retrieve_past_case_optimized(
            state=state,
            summary_vectordb=summary_vectordb,
            k=k_cases,
            metadata_threshold=metadata_threshold,
            similarity_threshold=similarity_threshold,
        )
    )

    # Wrap the simplified case content retrieval function - exactly like your notebook
    retrieve_simplified_case_content_with_dir = (
        lambda state: retrieve_simplified_case_content(
            state=state, case_json_dir=case_json_dir
        )
    )

    # Add nodes with optimized functions - exactly like your notebook
    graph.add_node(
        "process_case_summary", lambda state: process_case_summary(state, llm)
    )
    graph.add_node(
        "optimized_guideline_retrieval",
        lambda state: optimized_guideline_retrieval_decision(state, llm),
    )
    graph.add_node(
        "guideline_retrieve_content",
        lambda state: retrieved_guideline_section_content(state, full_guidelines),
    )
    graph.add_node(
        "attending_specialty_decision",
        lambda state: attending_specialty_decision(state, llm),
    )
    graph.add_node(
        "retrieve_past_case_optimized", retrieve_past_case_optimized_with_params
    )
    graph.add_node(
        "retrieve_simplified_case_content", retrieve_simplified_case_content_with_dir
    )
    graph.add_node(
        "simplified_predict_category",
        lambda state: simplified_triage_prediction(state, llm),
    )

    # Define edges - exactly like your notebook
    graph.add_edge("process_case_summary", "optimized_guideline_retrieval")
    graph.add_edge("optimized_guideline_retrieval", "guideline_retrieve_content")
    graph.add_edge("guideline_retrieve_content", "attending_specialty_decision")
    graph.add_edge("attending_specialty_decision", "retrieve_past_case_optimized")
    graph.add_edge("retrieve_past_case_optimized", "retrieve_simplified_case_content")
    graph.add_edge("retrieve_simplified_case_content", "simplified_predict_category")
    graph.add_edge("simplified_predict_category", END)

    # Set the entry point
    graph.set_entry_point("process_case_summary")

    # Compile and return the graph
    return graph.compile()


def create_initial_state(
    case_json_full: Dict[str, Any],
    guideline_section_metadata: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create the initial state for graph execution.

    Args:
        case_json_full: Complete case JSON data
        guideline_section_metadata: Guideline metadata (optional, will load if not provided)

    Returns:
        Initialized state dictionary
    """
    # Load guideline metadata if not provided
    if guideline_section_metadata is None:
        full_guidelines = load_triage_guidelines()
        guideline_section_metadata = extract_guideline_metadata(full_guidelines)

    # Create initial state exactly like in the notebook
    initial_state = {
        "case_json_full": case_json_full,
        "guideline_section_metadata": guideline_section_metadata,
        "specialty_list": AGENT_SPECIALTY_LIST,
    }

    print(f"DEBUG: Created initial state with keys: {list(initial_state.keys())}")
    print(
        f"DEBUG: case_json_full in initial state: {initial_state['case_json_full'] is not None}"
    )

    return initial_state


def validate_graph_inputs(
    llm, summary_vectordb: Chroma, case_json_dir: str, case_json_full: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate all inputs required for graph execution.

    Args:
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing case JSON files
        case_json_full: Complete case JSON data

    Returns:
        Validation result dictionary
    """
    validation = {"valid": True, "errors": [], "warnings": []}

    # Validate LLM
    if llm is None:
        validation["errors"].append("LLM instance is required")
        validation["valid"] = False

    # Validate vector database
    try:
        doc_count = summary_vectordb._collection.count()
        if doc_count == 0:
            validation["warnings"].append("Vector database is empty")
    except Exception as e:
        validation["errors"].append(f"Vector database error: {str(e)}")
        validation["valid"] = False

    # Validate case JSON directory
    import os

    if not os.path.exists(case_json_dir):
        validation["errors"].append(
            f"Case JSON directory does not exist: {case_json_dir}"
        )
        validation["valid"] = False

    # Validate case data
    if not case_json_full:
        validation["errors"].append("Case JSON data is empty")
        validation["valid"] = False
    else:
        # Check for required sections
        required_sections = ["Demographics", "Clinical Presentation"]
        for section in required_sections:
            if section not in case_json_full:
                validation["warnings"].append(
                    f"Missing section in case data: {section}"
                )

    # Validate guidelines
    try:
        guidelines = load_triage_guidelines()
        if not guidelines:
            validation["warnings"].append("No triage guidelines loaded")
    except Exception as e:
        validation["errors"].append(f"Guidelines loading error: {str(e)}")
        validation["valid"] = False

    return validation


def get_graph_info(graph) -> Dict[str, Any]:
    """
    Get information about the constructed graph.

    Args:
        graph: Compiled LangGraph

    Returns:
        Dictionary with graph information
    """
    try:
        # Get basic graph structure
        nodes = list(graph.nodes.keys()) if hasattr(graph, "nodes") else []

        info = {
            "node_count": len(nodes),
            "nodes": nodes,
            "entry_point": "process_case_summary",
            "end_point": "END",
            "pipeline_stages": [
                "Case Preprocessing",
                "Guideline Retrieval",
                "Specialty Prediction",
                "Past Case Retrieval",
                "Final Assessment",
            ],
        }

        return info

    except Exception as e:
        return {"error": str(e)}


def execute_graph_with_validation(
    graph, initial_state: GraphState, validate_inputs: bool = True
) -> Dict[str, Any]:
    """
    Execute the graph with optional input validation and error handling.

    Args:
        graph: Compiled LangGraph
        initial_state: Initial state for execution
        validate_inputs: Whether to validate inputs before execution

    Returns:
        Execution result with state and any errors
    """
    result = {"success": False, "final_state": None, "errors": [], "execution_info": {}}

    try:
        # Optional input validation
        if validate_inputs:
            state_validation = initial_state.validate()
            if state_validation:
                result["warnings"] = state_validation

        # Execute the graph
        final_state = graph.invoke(initial_state)

        result["success"] = True
        result["final_state"] = final_state
        result["execution_info"] = {
            "nodes_executed": len(graph.nodes) if hasattr(graph, "nodes") else 0,
            "has_final_assessment": final_state.get("final_assessment") is not None,
        }

    except Exception as e:
        result["errors"].append(f"Graph execution error: {str(e)}")
        result["execution_info"]["error_details"] = str(e)

    return result
