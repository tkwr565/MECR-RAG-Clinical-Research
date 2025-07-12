"""
Case retrieval module for the Emergency Medicine Triage RAG System.

This module handles the retrieval of similar past cases using vector similarity search
with metadata filtering based on demographics and specialty predictions.
"""

import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma

from ..data.state import GraphState
from ..config.specialty_mapping import get_standardized_specialty
from ..models.embeddings import get_default_embeddings
from ..retrieval.utils import (
    get_filtered_document_ids,
    calculate_similarity_score,
    format_retrieved_case_result,
    simplify_case_content,
)


def retrieve_past_cases_optimized(
    summary_text: str,
    case_json_full: Dict[str, Any],
    selected_attending_specialty: Dict[str, Any],
    summary_vectordb: Chroma,
    k: int = 5,
    metadata_threshold: float = 1.0,
    similarity_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Retrieve similar past cases with stricter similarity thresholds and metadata filtering.

    Args:
        summary_text: Clinical summary for vector search
        case_json_full: Complete case JSON data
        selected_attending_specialty: Predicted specialty information
        summary_vectordb: Vector database for case summaries
        k: Number of cases to retrieve
        metadata_threshold: Threshold for metadata matching
        similarity_threshold: Minimum similarity score to include a case

    Returns:
        List of retrieved cases with metadata and similarity scores
    """
    # Get filtered document IDs based on metadata
    filtered_ids = get_filtered_document_ids(
        vectordb=summary_vectordb,
        case_json_full=case_json_full,
        selected_attending_specialty=selected_attending_specialty,
        metadata_threshold=metadata_threshold,
    )

    print(f"Found {len(filtered_ids)} unique cases after filtering")

    if not filtered_ids:
        print("No cases found after filtering, returning empty list")
        return []

    # Perform vector similarity search
    embeddings = get_default_embeddings()
    summary_embedding = embeddings.embed_query(summary_text)

    # Search with increased k to allow for filtering by similarity threshold
    search_k = min(k * 2, len(filtered_ids))

    summary_results = (
        summary_vectordb.similarity_search_by_vector_with_relevance_scores(
            embedding=summary_embedding,
            k=search_k,
            filter={"doc_id": {"$in": filtered_ids}},
        )
    )

    # Apply similarity threshold and limit results
    final_results = []
    for doc, score in summary_results:
        similarity = calculate_similarity_score(score)

        # Apply stricter similarity threshold
        if similarity < similarity_threshold:
            print(
                f"Filtering out case {doc.metadata['doc_id']} with similarity {similarity:.3f} < {similarity_threshold}"
            )
            continue

        # Stop when we have enough high-quality cases
        if len(final_results) >= k:
            break

        # Get standardized specialty
        std_specialty = None
        if "Attending Specialty" in doc.metadata:
            dataset_specialty = doc.metadata["Attending Specialty"]
            std_specialty = get_standardized_specialty(dataset_specialty)

        # Format result
        result = format_retrieved_case_result(
            doc=doc, similarity_score=similarity, standardized_specialty=std_specialty
        )

        final_results.append(result)

    print(
        f"========= Retrieved {len(final_results)} high-quality past cases (similarity >= {similarity_threshold}) ==============="
    )
    return final_results


def retrieve_past_case_optimized(
    state: GraphState,
    summary_vectordb: Chroma,
    k: int = 5,
    metadata_threshold: float = 1.0,
    similarity_threshold: float = 0.7,
) -> GraphState:
    """
    Retrieve similar past cases (GraphState node function).

    Args:
        state: Current GraphState with case data and specialty prediction
        summary_vectordb: Vector database for case summaries
        k: Number of cases to retrieve
        metadata_threshold: Threshold for metadata matching
        similarity_threshold: Minimum similarity score to include a case

    Returns:
        Updated state with retrieved_cases
    """
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        summary_text = state.get("summary_text")
        case_json_full = state.get("case_json_full", {})
        selected_attending_specialty = state.get("selected_attending_specialty", {})
    else:
        summary_text = state.summary_text
        case_json_full = state.case_json_full
        selected_attending_specialty = state.selected_attending_specialty

    retrieved_cases = retrieve_past_cases_optimized(
        summary_text=summary_text,
        case_json_full=case_json_full,
        selected_attending_specialty=selected_attending_specialty,
        summary_vectordb=summary_vectordb,
        k=k,
        metadata_threshold=metadata_threshold,
        similarity_threshold=similarity_threshold,
    )

    # Update state - handle both dict and GraphState
    if isinstance(state, dict):
        updated_state = state.copy()
        updated_state["retrieved_cases"] = retrieved_cases
    else:
        updated_state = state.copy()
        updated_state.retrieved_cases = retrieved_cases

    return updated_state


def retrieve_case_content_from_json(
    retrieved_cases: List[Dict[str, Any]], case_json_dir: str, simplify: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve full JSON content for each retrieved case.

    Args:
        retrieved_cases: List of retrieved case metadata
        case_json_dir: Directory containing the full JSON case files
        simplify: Whether to simplify case content (remove detailed sections)

    Returns:
        List of cases with full content
    """
    case_contents = []

    print(f"Found {len(retrieved_cases)} cases to retrieve content for")

    for case in retrieved_cases:
        doc_id = case["doc_id"]
        json_filename = f"{doc_id}.json"
        json_path = os.path.join(case_json_dir, json_filename)

        try:
            # Attempt to read the JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                case_json = json.load(f)

            # Simplify or keep full content
            if simplify:
                content = simplify_case_content(case_json)
            else:
                content = case_json

            # Add to our list with metadata for reference
            case_contents.append(
                {
                    "doc_id": doc_id,
                    "original_filename": case.get("original_filename", ""),
                    "similarity_score": case.get("similarity_score", 0.0),
                    "content": content,
                }
            )
            print(f"Successfully loaded content for {doc_id}")

        except FileNotFoundError:
            print(
                f"Warning: JSON file not found for doc_id: {doc_id} at path: {json_path}"
            )
            case_contents.append(
                {
                    "doc_id": doc_id,
                    "original_filename": case.get("original_filename", ""),
                    "similarity_score": case.get("similarity_score", 0.0),
                    "content": None,
                    "error": "File not found",
                }
            )

        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON format in file for doc_id: {doc_id}")
            case_contents.append(
                {
                    "doc_id": doc_id,
                    "original_filename": case.get("original_filename", ""),
                    "similarity_score": case.get("similarity_score", 0.0),
                    "content": None,
                    "error": "Invalid JSON format",
                }
            )

    print("=" * 80)
    print(f"Retrieved content for {len(case_contents)} cases")
    return case_contents


def retrieve_simplified_case_content(
    state: GraphState, case_json_dir: str
) -> GraphState:
    """
    Retrieve simplified JSON content for each retrieved case (GraphState node function).

    Args:
        state: Current state with retrieved_cases list
        case_json_dir: Directory containing the full JSON case files

    Returns:
        Updated state with simplified retrieved_cases_content list
    """
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        retrieved_cases = state.get("retrieved_cases", [])
    else:
        retrieved_cases = state.retrieved_cases

    if not retrieved_cases:
        print("No retrieved cases found in state")
        simplified_case_contents = []
    else:
        simplified_case_contents = retrieve_case_content_from_json(
            retrieved_cases=retrieved_cases, case_json_dir=case_json_dir, simplify=True
        )

    # Update state - handle both dict and GraphState
    if isinstance(state, dict):
        updated_state = state.copy()
        updated_state["retrieved_cases_content"] = simplified_case_contents
    else:
        updated_state = state.copy()
        updated_state.retrieved_cases_content = simplified_case_contents

    return updated_state


def format_past_cases_context(retrieved_cases_content: List[Dict[str, Any]]) -> str:
    """
    Format retrieved past cases into context string for prompts.

    Args:
        retrieved_cases_content: List of cases with content

    Returns:
        Formatted context string
    """
    if not retrieved_cases_content:
        return "No similar past cases found."

    context = ""
    for i, case in enumerate(retrieved_cases_content):
        if case.get("content"):
            content = case["content"]
            context += f"Case {i+1}: {content.get('Demographics', {})}, "
            context += f"Summary: {content.get('Clinical Summary', '')}, "
            context += f"Category: {content.get('Case Disposition', {}).get('Triage Category', '')}\n\n"

    return context


def validate_retrieved_cases(
    retrieved_cases: List[Dict[str, Any]],
    min_similarity: float = 0.5,
    max_cases: int = 10,
) -> Dict[str, Any]:
    """
    Validate retrieved cases for quality and completeness.

    Args:
        retrieved_cases: List of retrieved cases
        min_similarity: Minimum expected similarity score
        max_cases: Maximum expected number of cases

    Returns:
        Validation result dictionary
    """
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "case_count": len(retrieved_cases),
        "avg_similarity": 0.0,
        "min_similarity": 1.0,
        "max_similarity": 0.0,
    }

    if not retrieved_cases:
        validation["warnings"].append("No cases retrieved")
        return validation

    # Calculate similarity statistics
    similarities = [case.get("similarity_score", 0.0) for case in retrieved_cases]
    validation["avg_similarity"] = sum(similarities) / len(similarities)
    validation["min_similarity"] = min(similarities)
    validation["max_similarity"] = max(similarities)

    # Check similarity thresholds
    low_similarity_cases = [s for s in similarities if s < min_similarity]
    if low_similarity_cases:
        validation["warnings"].append(
            f"{len(low_similarity_cases)} cases below similarity threshold {min_similarity}"
        )

    # Check case count
    if len(retrieved_cases) > max_cases:
        validation["warnings"].append(
            f"Retrieved {len(retrieved_cases)} cases, more than expected {max_cases}"
        )

    # Check for required fields
    for i, case in enumerate(retrieved_cases):
        required_fields = ["doc_id", "similarity_score"]
        missing_fields = [field for field in required_fields if field not in case]
        if missing_fields:
            validation["errors"].append(f"Case {i}: missing fields {missing_fields}")
            validation["valid"] = False

    return validation


def get_case_retrieval_statistics(
    retrievals: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Calculate statistics for multiple case retrievals.

    Args:
        retrievals: List of retrieval results (each is a list of cases)

    Returns:
        Dictionary with retrieval statistics
    """
    if not retrievals:
        return {"total_retrievals": 0}

    stats = {
        "total_retrievals": len(retrievals),
        "avg_cases_per_retrieval": 0.0,
        "avg_similarity": 0.0,
        "min_similarity": 1.0,
        "max_similarity": 0.0,
        "empty_retrievals": 0,
        "specialty_distribution": {},
    }

    total_cases = 0
    all_similarities = []

    for retrieval in retrievals:
        if not retrieval:
            stats["empty_retrievals"] += 1
            continue

        total_cases += len(retrieval)

        for case in retrieval:
            # Collect similarity scores
            similarity = case.get("similarity_score", 0.0)
            all_similarities.append(similarity)

            # Count specialties
            specialty = case.get("standardized_specialty", "Unknown")
            stats["specialty_distribution"][specialty] = (
                stats["specialty_distribution"].get(specialty, 0) + 1
            )

    if retrievals:
        stats["avg_cases_per_retrieval"] = total_cases / len(retrievals)

    if all_similarities:
        stats["avg_similarity"] = sum(all_similarities) / len(all_similarities)
        stats["min_similarity"] = min(all_similarities)
        stats["max_similarity"] = max(all_similarities)

    return stats
