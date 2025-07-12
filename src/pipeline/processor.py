"""
Pipeline processor module for the Emergency Medicine Triage RAG System.

This module provides high-level functions for processing single cases and batch operations,
coordinating the complete triage assessment pipeline from input to output.
"""

import os
import json
import traceback
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..data.loaders import (
    load_case_from_json,
    load_triage_guidelines,
    extract_guideline_metadata,
)
from ..data.state import GraphState
from ..config.specialty_mapping import AGENT_SPECIALTY_LIST
from ..pipeline.graph_builder import build_optimized_triage_graph, create_initial_state


def process_case_file_optimized(
    file_path: str,
    llm,
    summary_vectordb,
    case_json_dir: str,
    k_cases: int = 5,
    metadata_threshold: float = 1.0,
    similarity_threshold: float = 0.7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a case file using the optimized triage assessment pipeline.

    Args:
        file_path: Path to the case file (json)
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing case JSON files
        k_cases: Number of past cases to retrieve
        metadata_threshold: Threshold for metadata matching
        similarity_threshold: Minimum similarity score for past cases
        verbose: Whether to print processing information

    Returns:
        Dict[str, Any]: Complete state from the optimized triage graph execution
    """
    try:
        # Load and format the case
        case_data = load_case_from_json(file_path)

        if verbose:
            print("Case Content:")
            print(f"Age: {case_data.get('Demographics', {}).get('Age', 'Unknown')}")
            print(f"Sex: {case_data.get('Demographics', {}).get('Sex', 'Unknown')}")
            chief_complaint = case_data.get("Clinical Presentation", {}).get(
                "Chief complaint", "Unknown"
            )
            print(f"Chief Complaint: {chief_complaint[:100]}...")
            print("-" * 80)

        # Build the optimized triage graph
        triage_graph = build_optimized_triage_graph(
            llm=llm,
            summary_vectordb=summary_vectordb,
            case_json_dir=case_json_dir,
            k_cases=k_cases,
            metadata_threshold=metadata_threshold,
            similarity_threshold=similarity_threshold,
        )

        # Create initial state
        initial_state = create_initial_state(case_json_full=case_data)

        # Execute the graph
        final_state = triage_graph.invoke(initial_state)

        return final_state

    except Exception as e:
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "file_path": file_path,
        }
        print(f"Error processing case file {file_path}: {str(e)}")
        if verbose:
            print(traceback.format_exc())
        return error_info


def process_case_data_optimized(
    case_data: Dict[str, Any],
    llm,
    summary_vectordb,
    case_json_dir: str,
    k_cases: int = 5,
    metadata_threshold: float = 1.0,
    similarity_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Process case data (already loaded) using the optimized triage assessment pipeline.

    Args:
        case_data: Case JSON data
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing case JSON files
        k_cases: Number of past cases to retrieve
        metadata_threshold: Threshold for metadata matching
        similarity_threshold: Minimum similarity score for past cases

    Returns:
        Dict[str, Any]: Complete state from the optimized triage graph execution
    """
    try:
        # Build the optimized triage graph
        triage_graph = build_optimized_triage_graph(
            llm=llm,
            summary_vectordb=summary_vectordb,
            case_json_dir=case_json_dir,
            k_cases=k_cases,
            metadata_threshold=metadata_threshold,
            similarity_threshold=similarity_threshold,
        )

        # Create initial state
        initial_state = create_initial_state(case_json_full=case_data)

        # Execute the graph
        final_state = triage_graph.invoke(initial_state)

        return final_state

    except Exception as e:
        error_info = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error processing case data: {str(e)}")
        return error_info


def batch_process_directory_optimized(
    input_directory: str,
    output_directory: str,
    llm,
    summary_vectordb,
    case_json_dir: str,
    max_files: Optional[int] = None,
    k_cases: int = 5,
    metadata_threshold: float = 1.0,
    similarity_threshold: float = 0.7,
    verbose: bool = True,
) -> int:
    """
    Process all JSON files using the optimized pipeline and save simplified results.

    Args:
        input_directory: Directory containing input JSON files
        output_directory: Directory to save results
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing case JSON files
        max_files: Maximum number of files to process (None for all)
        k_cases: Number of past cases to retrieve
        metadata_threshold: Threshold for metadata matching
        similarity_threshold: Minimum similarity score for past cases
        verbose: Whether to print detailed progress

    Returns:
        Number of files processed successfully
    """
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Find all JSON files
    json_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    if max_files:
        json_files = json_files[:max_files]

    print(f"Found {len(json_files)} JSON files to process with optimized pipeline")

    # Track statistics
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Process each file
    for file_path in tqdm(json_files, desc="Processing cases (optimized)"):
        try:
            # Get relative path for output structure
            rel_path = os.path.relpath(file_path, input_directory)
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]

            # Create output subdirectory
            rel_dir = os.path.dirname(rel_path)
            output_subdir = os.path.join(output_directory, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)

            # Check if output file already exists
            output_file_path = os.path.join(
                output_subdir, f"{base_name}_optimized_output.json"
            )
            if os.path.exists(output_file_path):
                if verbose:
                    print(
                        f"\nSkipping {rel_path} - optimized output file already exists"
                    )
                skipped_count += 1
                continue

            if verbose:
                print(f"\nProcessing with optimized pipeline: {rel_path}")

            # Process the case with optimized pipeline
            state = process_case_file_optimized(
                file_path=file_path,
                llm=llm,
                summary_vectordb=summary_vectordb,
                case_json_dir=case_json_dir,
                k_cases=k_cases,
                metadata_threshold=metadata_threshold,
                similarity_threshold=similarity_threshold,
                verbose=False,  # Reduce verbosity in batch mode
            )

            # Check if processing was successful
            if "error" in state:
                print(f"Error processing {rel_path}: {state['error']}")
                error_count += 1
                continue

            # Save simplified output data
            output_data = create_output_summary(
                state=state, k_cases=k_cases, similarity_threshold=similarity_threshold
            )

            # Save the output
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            processed_count += 1

            if verbose:
                print(f"Saved optimized output to: {output_file_path}")
                print(
                    f"Guidelines retrieved: {len(state.get('selected_guideline_sections', []))}"
                )
                print(
                    f"Past cases retrieved: {len(state.get('retrieved_cases_content', []))}"
                )

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            if verbose:
                print(traceback.format_exc())
            error_count += 1

    # Print summary
    print("\nOptimized Processing Complete")
    print(f"Total files found: {len(json_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped (already exist): {skipped_count}")
    print(f"Files with errors: {error_count}")

    return processed_count


def create_output_summary(
    state: Dict[str, Any], k_cases: int, similarity_threshold: float
) -> Dict[str, Any]:
    """
    Create a summary output from the pipeline state.

    Args:
        state: Final state from pipeline execution
        k_cases: Number of cases retrieved
        similarity_threshold: Similarity threshold used

    Returns:
        Simplified output data dictionary
    """
    return {
        "summary_text": state.get("summary_text"),
        "case_json_full": state.get("case_json_full"),
        "selected_guideline_sections": state.get("selected_guideline_sections", []),
        "guideline_count": len(state.get("selected_guideline_sections", [])),
        "retrieved_cases_simplified": [
            {
                "doc_id": case.get("doc_id"),
                "similarity_score": case.get("similarity_score"),
                "content": case.get("content"),
            }
            for case in state.get("retrieved_cases_content", [])
        ],
        "past_cases_count": len(state.get("retrieved_cases_content", [])),
        "final_assessment": state.get("final_assessment"),
        "optimization_info": {
            "max_guidelines": 2,
            "max_past_cases": k_cases,
            "similarity_threshold": similarity_threshold,
            "reasoning_steps": 3,
        },
    }


def create_enhanced_output_summary(
    state: Dict[str, Any],
    k_cases: int,
    similarity_threshold: float,
    extracted_info: Dict[str, Any],
    processing_time: float,
) -> Dict[str, Any]:
    """
    Create a comprehensive output summary matching the desired API response format.

    Args:
        state: Final state from pipeline execution
        k_cases: Number of cases retrieved
        similarity_threshold: Similarity threshold used
        extracted_info: Extracted category, confidence, and rationale
        processing_time: Total processing time in seconds

    Returns:
        Comprehensive output data dictionary
    """

    # Helper function to expand case with all possible fields
    def expand_case_with_defaults(case_json: Dict[str, Any]) -> Dict[str, Any]:
        """Expand case JSON with all possible empty fields for comprehensive structure"""
        expanded_case = {
            "Demographics": {
                "Sex": case_json.get("Demographics", {}).get("Sex", ""),
                "Age": case_json.get("Demographics", {}).get("Age", ""),
                "Ambulatory status": case_json.get("Demographics", {}).get(
                    "Ambulatory status", ""
                ),
                "Risk of fall": case_json.get("Demographics", {}).get(
                    "Risk of fall", ""
                ),
                "Informant": case_json.get("Demographics", {}).get("Informant", ""),
                "Communication": case_json.get("Demographics", {}).get(
                    "Communication", ""
                ),
                "Allergies": case_json.get("Demographics", {}).get("Allergies", ""),
                "ADR": case_json.get("Demographics", {}).get("ADR", ""),
                "Alerts": case_json.get("Demographics", {}).get("Alerts", ""),
                "Past health": case_json.get("Demographics", {}).get("Past health", ""),
                "TOCC": case_json.get("Demographics", {}).get("TOCC", ""),
                "Referral (if any)": case_json.get("Demographics", {}).get(
                    "Referral (if any)", ""
                ),
            },
            "Clinical Presentation": {
                "Chief complaint": case_json.get("Clinical Presentation", {}).get(
                    "Chief complaint",
                    case_json.get("Clinical_Presentation", {}).get(
                        "Chief_complaint", ""
                    ),
                ),
                "Condition on Arrival": case_json.get("Clinical Presentation", {}).get(
                    "Condition on Arrival",
                    case_json.get("Clinical_Presentation", {}).get(
                        "Condition_on_Arrival", ""
                    ),
                ),
            },
            "Vitals and Observations": {
                "Vital Signs": {
                    "GCS": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "GCS",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("GCS", ""),
                    ),
                    "BP": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "BP",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("BP", ""),
                    ),
                    "PR/AR": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "PR/AR",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("PR/AR", ""),
                    ),
                    "Temp": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "Temp",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("Temp", ""),
                    ),
                    "SpO2": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "SpO2",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("SpO2", ""),
                    ),
                    "RR": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "RR",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("RR", ""),
                    ),
                    "PFR": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "PFR",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("PFR", ""),
                    ),
                    "Limbs": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "Limbs",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("Limbs", ""),
                    ),
                    "Pupil": case_json.get("Vitals and Observations", {})
                    .get("Vital Signs", {})
                    .get(
                        "Pupil",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Vital_Signs", {})
                        .get("Pupil", ""),
                    ),
                },
                "Triage Intervention": {
                    "H'stix": case_json.get("Vitals and Observations", {})
                    .get("Triage Intervention", {})
                    .get(
                        "H'stix",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Triage_Intervention", {})
                        .get("H'stix", ""),
                    ),
                    "H'cue": case_json.get("Vitals and Observations", {})
                    .get("Triage Intervention", {})
                    .get(
                        "H'cue",
                        case_json.get("Vitals_and_Observations", {})
                        .get("Triage_Intervention", {})
                        .get("H'cue", ""),
                    ),
                },
                "Menstruation / Pregnancy Status": case_json.get(
                    "Vitals and Observations", {}
                ).get(
                    "Menstruation / Pregnancy Status",
                    case_json.get("Vitals_and_Observations", {}).get(
                        "Menstruation / Pregnancy Status", ""
                    ),
                ),
            },
        }
        return expanded_case

    # Process retrieved cases to include full content (Clinical Summary + Case Disposition)
    retrieved_cases_simplified = []
    for case in state.get("retrieved_cases_content", []):
        if case.get("content"):
            content = case["content"]

            # Extract Clinical Summary and Case Disposition
            case_content = {
                "Clinical Summary": content.get("Clinical Summary", ""),
                "Case Disposition": {
                    "Triage Category": content.get("Case Disposition", {}).get(
                        "Triage Category", ""
                    ),
                    "Diagnosis": content.get("Case Disposition", {}).get(
                        "Diagnosis", ""
                    ),
                    "Attending Specialty": content.get("Case Disposition", {}).get(
                        "Attending Specialty", ""
                    ),
                    "Trauma Type": content.get("Case Disposition", {}).get(
                        "Trauma Type", ""
                    ),
                    "Discharge Destination": content.get("Case Disposition", {}).get(
                        "Discharge Destination", ""
                    ),
                },
            }

            retrieved_cases_simplified.append(
                {
                    "doc_id": case.get("doc_id", ""),
                    "similarity_score": case.get("similarity_score", 0.0),
                    "content": case_content,
                }
            )

    return {
        "summary_text": state.get("summary_text", ""),
        "case_json_full": expand_case_with_defaults(state.get("case_json_full", {})),
        "selected_guideline_sections": [
            section.get("section_title", "")
            for section in state.get("retrieved_guideline_content", [])
        ],
        "guideline_count": len(state.get("retrieved_guideline_content", [])),
        "retrieved_cases_simplified": retrieved_cases_simplified,
        "final_result": {
            "job_id": "",  # Will be filled by the API endpoint
            "status": "completed",
            "category": extracted_info.get("category"),
            "confidence": extracted_info.get("confidence"),
            "rationale": extracted_info.get("rationale"),
            "full_response": state.get("final_assessment", ""),  # Full LLM response
            "processing_time_seconds": round(processing_time, 2),
            "error_message": None,
            "request_id": "",  # Will be filled by the API endpoint
        },
    }


def validate_pipeline_setup(
    llm, summary_vectordb, case_json_dir: str
) -> Dict[str, Any]:
    """
    Validate that all pipeline components are properly configured.

    Args:
        llm: Language model instance
        summary_vectordb: Vector database for past cases
        case_json_dir: Directory containing case JSON files

    Returns:
        Validation result dictionary
    """
    validation = {"valid": True, "errors": [], "warnings": [], "component_status": {}}

    # Check LLM
    try:
        if llm is None:
            validation["errors"].append("LLM instance is None")
            validation["valid"] = False
        validation["component_status"]["llm"] = "available" if llm else "missing"
    except Exception as e:
        validation["errors"].append(f"LLM validation error: {str(e)}")
        validation["valid"] = False

    # Check vector database
    try:
        doc_count = summary_vectordb._collection.count()
        validation["component_status"][
            "vector_db"
        ] = f"available ({doc_count} documents)"
        if doc_count == 0:
            validation["warnings"].append("Vector database is empty")
    except Exception as e:
        validation["errors"].append(f"Vector database error: {str(e)}")
        validation["valid"] = False
        validation["component_status"]["vector_db"] = "error"

    # Check case JSON directory
    try:
        if not os.path.exists(case_json_dir):
            validation["errors"].append(
                f"Case JSON directory does not exist: {case_json_dir}"
            )
            validation["valid"] = False
            validation["component_status"]["case_json_dir"] = "missing"
        else:
            json_files = [f for f in os.listdir(case_json_dir) if f.endswith(".json")]
            validation["component_status"][
                "case_json_dir"
            ] = f"available ({len(json_files)} JSON files)"
    except Exception as e:
        validation["errors"].append(f"Case JSON directory error: {str(e)}")
        validation["valid"] = False
        validation["component_status"]["case_json_dir"] = "error"

    # Check guidelines
    try:
        guidelines = load_triage_guidelines()
        validation["component_status"][
            "guidelines"
        ] = f"available ({len(guidelines)} sections)"
    except Exception as e:
        validation["errors"].append(f"Guidelines loading error: {str(e)}")
        validation["valid"] = False
        validation["component_status"]["guidelines"] = "error"

    return validation


def get_pipeline_info() -> Dict[str, Any]:
    """
    Get information about the pipeline configuration and capabilities.

    Returns:
        Dictionary with pipeline information
    """
    return {
        "pipeline_name": "Emergency Medicine Triage RAG System",
        "version": "optimized",
        "stages": [
            "Case Preprocessing (Summarization)",
            "Guideline Retrieval (Max 2 sections)",
            "Specialty Prediction",
            "Past Case Retrieval (Metadata + Similarity filtering)",
            "Final Triage Assessment (3-step reasoning)",
        ],
        "default_parameters": {
            "k_cases": 5,
            "metadata_threshold": 1.0,
            "similarity_threshold": 0.7,
            "max_guideline_sections": 2,
        },
        "supported_models": ["deepseek", "gpt4o", "claude"],
        "input_format": "JSON case files",
        "output_format": "Structured triage assessment with reasoning",
    }
