"""
Data loading utilities for the Emergency Medicine Triage RAG System.

This module provides functions to load case data, triage guidelines,
and other required data files for the RAG pipeline.
"""

import json
import os
from typing import Dict, List, Any
from langchain_community.vectorstores import Chroma

from ..config.settings import settings
from ..models.embeddings import get_default_embeddings


def load_case_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load a FULL case from a JSON file as NEW case.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        Dict[str, Any]: FULL Case data, with "Case Disposition" removed (required for test/val set).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    # Remove sections if they exist, required for ground_truth set
    if "Case Disposition" in case_data:
        case_data.pop("Case Disposition")
    if "Ground Truth" in case_data:
        case_data.pop("Ground Truth")

    return case_data


def load_triage_guidelines(file_path: str = None) -> List[Dict[str, str]]:
    """
    Load the triage guidelines from a JSON file.

    Args:
        file_path (str, optional): Path to the JSON file.
                                 Defaults to settings.TRIAGE_GUIDELINES_PATH

    Returns:
        List[Dict[str, str]]: List of guideline sections with title, content, and summary
    """
    if file_path is None:
        file_path = settings.TRIAGE_GUIDELINES_PATH

    with open(file_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    print(f"Loaded {len(guidelines)} sections from {file_path}")
    return guidelines


def extract_guideline_metadata(
    guidelines: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Extract just the metadata (title and summary) for retrieval decisions.

    Args:
        guidelines: List of guideline sections with full content

    Returns:
        List of guideline metadata with only title and summary
    """
    return [
        {"section_title": section["section_title"], "summary": section["summary"]}
        for section in guidelines
    ]


def load_vector_database(model: str, base_dir: str = None) -> Chroma:
    """
    Load existing Chroma vector database for past cases.

    Args:
        model: Model name for database path
        base_dir: Base directory containing the vector databases (optional)

    Returns:
        Chroma VectorDB for summary text
    """
    if base_dir is None:
        base_dir = settings.get_vector_db_path(model)

    embeddings = get_default_embeddings()

    summary_vectordb = Chroma(
        persist_directory=os.path.join(base_dir, "summary_vectordb"),
        embedding_function=embeddings,
        collection_name="summary_text",
    )

    return summary_vectordb


def validate_vector_database(vectordb: Chroma) -> Dict[str, Any]:
    """
    Validate and get information about the vector database.

    Args:
        vectordb: Chroma vector database

    Returns:
        Dictionary with database information
    """
    try:
        doc_count = vectordb._collection.count()
        sample_ids = vectordb.get()["ids"][:3] if doc_count > 0 else []

        return {
            "status": "valid",
            "document_count": doc_count,
            "sample_ids": sample_ids,
            "collection_name": vectordb._collection.name,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "document_count": 0}


def get_case_json_directory(model: str) -> str:
    """
    Get the case JSON directory path for a specific model.

    Args:
        model: Model name

    Returns:
        Path to the case JSON directory
    """
    return settings.get_case_json_dir(model)


def load_case_json_from_id(
    doc_id: str, model: str, case_json_dir: str = None
) -> Dict[str, Any]:
    """
    Load a specific case JSON file by document ID.

    Args:
        doc_id: Document ID
        model: Model name for directory path
        case_json_dir: Directory containing JSON files (optional)

    Returns:
        Case JSON data

    Raises:
        FileNotFoundError: If the JSON file is not found
        json.JSONDecodeError: If the JSON file is invalid
    """
    if case_json_dir is None:
        case_json_dir = get_case_json_directory(model)

    json_filename = f"{doc_id}.json"
    json_path = os.path.join(case_json_dir, json_filename)

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_data_availability(model: str) -> Dict[str, Any]:
    """
    Check the availability of all required data files for a model.

    Args:
        model: Model name to check

    Returns:
        Dictionary with availability status
    """
    status = {
        "model": model,
        "guidelines_available": False,
        "vector_db_available": False,
        "case_json_dir_available": False,
        "errors": [],
    }

    # Check guidelines
    try:
        guidelines = load_triage_guidelines()
        status["guidelines_available"] = len(guidelines) > 0
        status["guidelines_count"] = len(guidelines)
    except Exception as e:
        status["errors"].append(f"Guidelines error: {str(e)}")

    # Check vector database
    try:
        vectordb = load_vector_database(model)
        db_info = validate_vector_database(vectordb)
        status["vector_db_available"] = db_info["status"] == "valid"
        status["vector_db_info"] = db_info
    except Exception as e:
        status["errors"].append(f"Vector DB error: {str(e)}")

    # Check case JSON directory
    try:
        case_json_dir = get_case_json_directory(model)
        status["case_json_dir_available"] = os.path.exists(case_json_dir)
        status["case_json_dir"] = case_json_dir
        if status["case_json_dir_available"]:
            json_files = [f for f in os.listdir(case_json_dir) if f.endswith(".json")]
            status["case_json_count"] = len(json_files)
    except Exception as e:
        status["errors"].append(f"Case JSON dir error: {str(e)}")

    status["all_available"] = (
        status["guidelines_available"]
        and status["vector_db_available"]
        and status["case_json_dir_available"]
    )

    return status


def get_available_models() -> List[str]:
    """
    Get list of models that have available data.

    Returns:
        List of model names with available data
    """
    from ..models.llm_factory import get_available_models

    model_info = get_available_models()
    available_models = []

    for model_key, info in model_info.items():
        db_name = info["db_name"]
        status = check_data_availability(db_name)
        if status["all_available"]:
            available_models.append(db_name)

    return available_models
