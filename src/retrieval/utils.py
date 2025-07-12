"""
Retrieval utility functions for the Emergency Medicine Triage RAG System.

This module contains utility functions for filtering documents by metadata,
extracting metadata from cases, and other retrieval-related operations.
"""

import os
import json
from typing import Dict, List, Any, Optional
from langchain_community.vectorstores import Chroma

from ..config.specialty_mapping import map_specialty_to_dataset_values


def filter_documents_by_metadata(
    vectordb: Chroma, metadata_filters: Dict[str, Any], threshold: float = 1.0
) -> List[str]:
    """
    Filter documents by metadata criteria and return doc_ids that meet the criteria.

    Args:
        vectordb: The vector database to search in
        metadata_filters: Dictionary of metadata filters to apply
        threshold: Minimum fraction of filters that must match (0.0-1.0)

    Returns:
        List of document IDs that match the metadata criteria
    """
    # Get all documents
    all_docs = vectordb.get()
    all_metadatas = all_docs["metadatas"]
    doc_ids = all_docs["ids"]

    matching_ids = []

    for i, metadata in enumerate(all_metadatas):
        # Count how many filters match
        match_count = 0
        total_applicable_filters = 0

        for key, value in metadata_filters.items():
            # Skip if the filter key doesn't exist in this document's metadata
            if key not in metadata:
                continue

            total_applicable_filters += 1

            # Check if values match (case-insensitive for string values)
            if isinstance(metadata[key], str) and isinstance(value, str):
                if metadata[key].lower() == value.lower():
                    match_count += 1
                    # print(f"Matched {key}: {metadata[key]} with {value}")
            else:
                if metadata[key] == value:
                    match_count += 1
                    # print(f"Matched {key}: {metadata[key]} with {value}")

        # Only include if at least threshold percentage of filters match
        if (
            total_applicable_filters > 0
            and (match_count / total_applicable_filters) >= threshold
        ):
            matching_ids.append(doc_ids[i])

    return matching_ids


def extract_age_group_from_case(case_json_full: Dict[str, Any]) -> str:
    """
    Extract age group (Adult/Paediatric) from case demographics.

    Args:
        case_json_full: Complete case JSON data

    Returns:
        Age group string ("Adult", "Paediatric", or "Unknown")
    """
    age_group = "Unknown"

    if "Demographics" in case_json_full and "Age" in case_json_full["Demographics"]:
        age_text = case_json_full["Demographics"]["Age"]

        if "years" in age_text.lower():
            try:
                age_years = int(age_text.lower().split("years")[0].strip())
                if age_years >= 18:
                    age_group = "Adult"
                else:
                    age_group = "Paediatric"
            except (ValueError, IndexError):
                pass
        elif "months" in age_text.lower() or "month" in age_text.lower():
            age_group = "Paediatric"

    return age_group


def create_metadata_filters(
    case_json_full: Dict[str, Any], selected_attending_specialty: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Create metadata filters for past case retrieval.

    Args:
        case_json_full: Complete case JSON data
        selected_attending_specialty: Predicted specialty information

    Returns:
        Dictionary with metadata filters including age group and specialty values
    """
    metadata = {}

    # Extract age group
    age_group = extract_age_group_from_case(case_json_full)
    if age_group != "Unknown":
        metadata["age_group"] = age_group

    # Extract specialty values
    specialty_values = []

    if selected_attending_specialty.get("primary_specialty"):
        primary_specialty = selected_attending_specialty["primary_specialty"]
        primary_raw_values = map_specialty_to_dataset_values(primary_specialty)
        specialty_values.extend(primary_raw_values)

    if selected_attending_specialty.get("secondary_specialty"):
        secondary_specialty = selected_attending_specialty["secondary_specialty"]
        secondary_raw_values = map_specialty_to_dataset_values(secondary_specialty)
        specialty_values.extend(secondary_raw_values)

    # Remove duplicates
    specialty_values = list(set(specialty_values))

    if specialty_values:
        metadata["specialty_values"] = specialty_values

    return metadata


def get_filtered_document_ids(
    vectordb: Chroma,
    case_json_full: Dict[str, Any],
    selected_attending_specialty: Dict[str, Any],
    metadata_threshold: float = 1.0,
) -> List[str]:
    """
    Get filtered document IDs based on case metadata and specialty.

    Args:
        vectordb: Vector database to search
        case_json_full: Complete case JSON data
        selected_attending_specialty: Predicted specialty information
        metadata_threshold: Threshold for metadata matching

    Returns:
        List of filtered document IDs
    """
    # Create base metadata filter (age group)
    base_metadata = {}
    age_group = extract_age_group_from_case(case_json_full)
    if age_group != "Unknown":
        base_metadata["age_group"] = age_group

    # Get specialty values
    specialty_values = []

    if selected_attending_specialty.get("primary_specialty"):
        primary_specialty = selected_attending_specialty["primary_specialty"]
        primary_raw_values = map_specialty_to_dataset_values(primary_specialty)
        specialty_values.extend(primary_raw_values)

    if selected_attending_specialty.get("secondary_specialty"):
        secondary_specialty = selected_attending_specialty["secondary_specialty"]
        secondary_raw_values = map_specialty_to_dataset_values(secondary_specialty)
        specialty_values.extend(secondary_raw_values)

    specialty_values = list(set(specialty_values))

    # Apply metadata filtering for each specialty
    all_filtered_ids = set()

    for specialty_value in specialty_values:
        current_filters = base_metadata.copy()
        current_filters["Attending Specialty"] = specialty_value

        filtered_ids = filter_documents_by_metadata(
            vectordb, current_filters, threshold=metadata_threshold
        )

        all_filtered_ids.update(filtered_ids)

    # If no specialty filters match, fall back to all documents
    if not all_filtered_ids:
        all_docs = vectordb.get()
        all_filtered_ids = set(all_docs["ids"])

    return list(all_filtered_ids)


def load_case_content_from_json(
    doc_id: str, case_json_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Load case content from JSON file by document ID.

    Args:
        doc_id: Document ID
        case_json_dir: Directory containing JSON files

    Returns:
        Case JSON content or None if not found/error
    """
    json_filename = f"{doc_id}.json"
    json_path = os.path.join(case_json_dir, json_filename)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading case {doc_id}: {e}")
        return None


def simplify_case_content(case_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify case content by removing detailed sections and keeping essentials.

    Args:
        case_json: Full case JSON data

    Returns:
        Simplified case content with demographics, summary, and disposition
    """
    simplified_case = case_json.copy()

    # Remove the detailed sections
    if "Clinical Presentation" in simplified_case:
        simplified_case.pop("Clinical Presentation")
    if "Vitals and Observations" in simplified_case:
        simplified_case.pop("Vitals and Observations")

    # Keep essential information: Demographics, Case Disposition, Summary
    essential_case = {
        "Demographics": simplified_case.get("Demographics", {}),
        "Clinical Summary": simplified_case.get("_summary_text", ""),
        "Case Disposition": simplified_case.get("Case Disposition", {}),
    }

    return essential_case


def calculate_similarity_score(distance: float) -> float:
    """
    Convert distance score to similarity score.

    Args:
        distance: Distance score from vector search

    Returns:
        Similarity score (1 - distance)
    """
    return 1 - distance


def format_retrieved_case_result(
    doc, similarity_score: float, standardized_specialty: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format a retrieved case result with metadata and content.

    Args:
        doc: Document from vector search
        similarity_score: Calculated similarity score
        standardized_specialty: Standardized specialty name

    Returns:
        Formatted case result dictionary
    """
    return {
        "doc_id": doc.metadata["doc_id"],
        "original_filename": doc.metadata.get("original_filename", ""),
        "similarity_score": similarity_score,
        "metadata": {
            k: v
            for k, v in doc.metadata.items()
            if k not in ["doc_id", "original_filename"]
        },
        "standardized_specialty": standardized_specialty,
        "content": doc.page_content,
    }


def validate_metadata_filters(metadata_filters: Dict[str, Any]) -> bool:
    """
    Validate metadata filters dictionary.

    Args:
        metadata_filters: Metadata filters to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(metadata_filters, dict):
        return False

    # Check that all values are simple types (str, int, float, bool)
    allowed_types = (str, int, float, bool)
    for key, value in metadata_filters.items():
        if not isinstance(key, str) or not isinstance(value, allowed_types):
            return False

    return True


def get_metadata_statistics(vectordb: Chroma) -> Dict[str, Any]:
    """
    Get statistics about metadata in the vector database.

    Args:
        vectordb: Vector database

    Returns:
        Dictionary with metadata statistics
    """
    try:
        all_docs = vectordb.get()
        all_metadatas = all_docs["metadatas"]

        # Count unique values for each metadata field
        field_stats = {}

        for metadata in all_metadatas:
            for key, value in metadata.items():
                if key not in field_stats:
                    field_stats[key] = set()
                field_stats[key].add(str(value))

        # Convert sets to counts
        stats = {
            "total_documents": len(all_docs["ids"]),
            "metadata_fields": {
                key: {
                    "unique_values": len(values),
                    "sample_values": list(values)[:5],  # First 5 unique values
                }
                for key, values in field_stats.items()
            },
        }

        return stats

    except Exception as e:
        return {"error": str(e), "total_documents": 0}
