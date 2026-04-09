#!/usr/bin/env python3
"""
Single Case Processing Script for Emergency Medicine Triage RAG System

This script processes a single case JSON file through the complete triage assessment
pipeline and outputs the results to console or file.

Usage:
    python scripts/run_single_case.py <case_file> --model <model_type> [options]

Example:
    python scripts/run_single_case.py data/test_case.json --model gpt4o --output results.json --verbose
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.llm_factory import create_llm, get_available_models
from src.data.loaders import load_vector_database, check_data_availability
from src.pipeline.processor import process_case_file_optimized, validate_pipeline_setup
from src.generation.triage_assessor import extract_final_category_from_assessment
from src.config.settings import settings
from src.utils.helpers import safe_json_save, get_file_info


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a single case through the Emergency Medicine Triage RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_single_case.py case.json --model gpt4o
  python scripts/run_single_case.py case.json --model deepseek --output results.json --verbose
  python scripts/run_single_case.py case.json --model claude --similarity-threshold 0.8 --k-cases 3
        """,
    )

    # Required arguments
    parser.add_argument("case_file", help="Path to the case JSON file to process")

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        choices=["deepseek", "gpt4o", "claude"],
        help="Model type to use for processing",
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o", help="Output file path (default: print to console)"
    )

    parser.add_argument(
        "--k-cases",
        type=int,
        default=settings.K_CASES,
        help=f"Number of past cases to retrieve (default: {settings.K_CASES})",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=settings.SIMILARITY_THRESHOLD,
        help=f"Minimum similarity threshold for past cases (default: {settings.SIMILARITY_THRESHOLD})",
    )

    parser.add_argument(
        "--metadata-threshold",
        type=float,
        default=settings.METADATA_THRESHOLD,
        help=f"Metadata matching threshold (default: {settings.METADATA_THRESHOLD})",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup without processing",
    )

    parser.add_argument(
        "--show-summary", action="store_true", help="Show extracted summary information"
    )

    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments and files."""
    errors = []

    # Check case file exists
    if not os.path.exists(args.case_file):
        errors.append(f"Case file not found: {args.case_file}")
    elif not args.case_file.endswith(".json"):
        errors.append(f"Case file must be a JSON file: {args.case_file}")

    # Check output directory exists if output specified
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            errors.append(f"Output directory does not exist: {output_dir}")

    # Validate parameter ranges
    if args.k_cases < 1 or args.k_cases > 20:
        errors.append("k-cases must be between 1 and 20")

    if not (0.0 <= args.similarity_threshold <= 1.0):
        errors.append("similarity-threshold must be between 0.0 and 1.0")

    if not (0.0 <= args.metadata_threshold <= 1.0):
        errors.append("metadata-threshold must be between 0.0 and 1.0")

    return errors


def setup_pipeline(model_type, verbose=False):
    """Setup the pipeline components."""
    if verbose:
        print(f"Setting up pipeline for model: {model_type}")

    try:
        # Initialize LLM
        llm = create_llm(model_type)
        if verbose:
            print("✓ LLM initialized")

        # Get model database name
        from src.models.llm_factory import get_model_name_for_db

        model_db_name = get_model_name_for_db(model_type)

        # Load vector database
        try:
            summary_vectordb = load_vector_database(model_db_name)
            if verbose:
                print("✓ Vector database loaded")
        except FileNotFoundError as e:
            print(f"❌ ERROR: Vector database not found for model '{model_type}'")
            print(f"Expected database: {model_db_name}")
            print(f"Details: {e}")
            print("\nPlease ensure the database files exist in the db/ directory:")
            print(f"- db/past_case/db_{model_db_name}_3000case/")
            return None, None, None, None
        except Exception as e:
            print(f"❌ ERROR: Failed to load vector database: {e}")
            return None, None, None, None

        # Get case JSON directory
        case_json_dir = settings.get_case_json_dir(model_db_name)
        if not os.path.exists(case_json_dir):
            print(f"❌ ERROR: Case JSON directory not found: {case_json_dir}")
            print(f"Please ensure past case data exists for model '{model_type}'")
            return None, None, None, None

        if verbose:
            print(f"✓ Case JSON directory: {case_json_dir}")

        return llm, summary_vectordb, case_json_dir, model_db_name

    except Exception as e:
        print(f"Error setting up pipeline: {e}")
        return None, None, None, None


def process_case(args, llm, summary_vectordb, case_json_dir):
    """Process the case file."""
    if args.verbose:
        print(f"\nProcessing case: {args.case_file}")
        print("-" * 50)

    try:
        # Process the case
        result = process_case_file_optimized(
            file_path=args.case_file,
            llm=llm,
            summary_vectordb=summary_vectordb,
            case_json_dir=case_json_dir,
            k_cases=args.k_cases,
            metadata_threshold=args.metadata_threshold,
            similarity_threshold=args.similarity_threshold,
            verbose=args.verbose,
        )

        # Check for errors
        if "error" in result:
            print(f"Error processing case: {result['error']}")
            return None

        return result

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


def display_results(result, args):
    """Display or save the results."""
    if not result:
        return False

    # Extract key information
    summary = result.get("summary_text", "N/A")
    guidelines = result.get("selected_guideline_sections", [])
    guideline_reasoning = result.get("guideline_reasoning", "N/A")
    past_cases_count = len(result.get("retrieved_cases_content", []))
    assessment = result.get("final_assessment", "N/A")

    # Handle both structured (dict) and legacy (string) assessment formats
    if isinstance(assessment, dict):
        # New structured format - direct access
        final_category = assessment.get("final_decision", {}).get("category", "N/A")
        confidence = assessment.get("final_decision", {}).get("confidence", "N/A")
        extracted_info = {
            "category": final_category,
            "confidence": confidence,
            "step1_category": assessment.get("step1_clinical_risk", {}).get("category", "N/A"),
            "step2_category": assessment.get("step2_guidelines", {}).get("category", "N/A"),
            "step3_category": assessment.get("step3_realworld_factors", {}).get("category", "N/A"),
            "rationale": assessment.get("final_decision", {}).get("reason", "N/A"),
        }
    else:
        # Legacy string format - use extraction function
        extracted_info = (
            extract_final_category_from_assessment(assessment)
            if assessment != "N/A"
            else {}
        )
        final_category = extracted_info.get("category", "N/A")
        confidence = extracted_info.get("confidence", "N/A")

    # Prepare output
    output_data = {
        "case_file": args.case_file,
        "processing_parameters": {
            "model": args.model,
            "k_cases": args.k_cases,
            "similarity_threshold": args.similarity_threshold,
            "metadata_threshold": args.metadata_threshold,
        },
        "results": {
            "clinical_summary": summary,
            "selected_guidelines": guidelines,
            "guideline_reasoning": guideline_reasoning,
            "guidelines_count": len(guidelines),
            "past_cases_retrieved": past_cases_count,
            "final_category": final_category,
            "confidence": confidence,
            "full_assessment": assessment,
            "structured_output": isinstance(assessment, dict),
        },
        "extracted_info": extracted_info,
    }

    # Show summary if requested
    if args.show_summary:
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Clinical Summary: {summary[:200]}...")
        print(f"Guidelines Retrieved: {len(guidelines)} - {guidelines}")
        if guideline_reasoning != "N/A":
            print(f"Guideline Reasoning: {guideline_reasoning}")
        print(f"Past Cases Retrieved: {past_cases_count}")
        print(f"\nTriage Assessment:")
        if isinstance(assessment, dict):
            print(f"  Step 1 (Clinical Risk): Category {extracted_info.get('step1_category', 'N/A')}")
            print(f"  Step 2 (Guidelines): Category {extracted_info.get('step2_category', 'N/A')}")
            print(f"  Step 3 (Real-world): Category {extracted_info.get('step3_category', 'N/A')}")
        print(f"  Final Triage Category: {final_category}")
        print(f"  Confidence: {confidence}")
        print("=" * 60)

    # Save or display results
    if args.output:
        success = safe_json_save(output_data, args.output)
        if success:
            print(f"\nResults saved to: {args.output}")
        else:
            print(f"\nError saving results to: {args.output}")
            return False
    else:
        print("\n" + "=" * 60)
        print("TRIAGE ASSESSMENT RESULTS")
        print("=" * 60)
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    return True


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Show file info if verbose
    if args.verbose:
        file_info = get_file_info(args.case_file)
        print(f"Case file info: {file_info}")

    # Setup pipeline
    llm, summary_vectordb, case_json_dir, model_db_name = setup_pipeline(
        args.model, args.verbose
    )

    if not all([llm, summary_vectordb, case_json_dir]):
        print("Failed to setup pipeline components")
        sys.exit(1)

    # Validate pipeline setup
    validation = validate_pipeline_setup(llm, summary_vectordb, case_json_dir)
    if not validation["valid"]:
        print("Pipeline validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        if validation["warnings"]:
            print("Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        if not args.validate_only:
            sys.exit(1)

    if args.validate_only:
        print("✓ Pipeline validation successful")
        print("Component status:")
        for component, status in validation["component_status"].items():
            print(f"  - {component}: {status}")
        return

    # Check data availability
    if args.verbose:
        data_status = check_data_availability(model_db_name)
        print(f"Data availability: {data_status['all_available']}")
        if data_status["errors"]:
            print("Data errors:")
            for error in data_status["errors"]:
                print(f"  - {error}")

    # Process the case
    result = process_case(args, llm, summary_vectordb, case_json_dir)

    if not result:
        print("Case processing failed")
        sys.exit(1)

    # Display results
    success = display_results(result, args)

    if success:
        print("\n✓ Case processing completed successfully")
    else:
        print("\n✗ Failed to save/display results")
        sys.exit(1)


if __name__ == "__main__":
    main()
