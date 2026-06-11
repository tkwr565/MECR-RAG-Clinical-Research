"""
Specialty prediction module for the Emergency Medicine Triage RAG System.

This module handles the prediction of attending medical specialties
based on case summaries and clinical presentations.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from ..data.state import GraphState
from ..models.schemas import create_specialty_prediction_schema


def create_specialty_prediction_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for attending specialty prediction.

    Note: This prompt is optimized for structured output and does not include
    JSON format examples (handled by Pydantic schema).

    Returns:
        ChatPromptTemplate for specialty prediction
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an emergency-medicine specialist assistant that helps determine the most appropriate attending specialty for presenting cases.

        Given a medical case summary, analyze the information and determine:
        1. The primary attending specialty that should handle the case
        2. A possible secondary attending specialty (if applicable)

        You must select specialties from the following list only:
        {specialties}

        Some clinical presentation may have overlapping specialty coverage. For example:
           - Injury-related cases could be covered by "Surgery", "Orthopaedics" or "Neurosurgery"
           - Paediatric cases could be covered by "Paediatrics" and other specialties
           If overlapping specialty is possible, include in "secondary_specialty"
        """,
            ),
            (
                "user",
                """
        Medical case summary:
        {summary_text}

        Based on the clinical presentation, which specialty should handle this case?
        """,
            ),
        ]
    )


def predict_attending_specialty(
    summary_text: str, specialty_list: list, llm
) -> Dict[str, Any]:
    """
    Predict the attending specialty for a given case summary using structured output.

    Args:
        summary_text: Clinical case summary
        specialty_list: List of available specialties
        llm: Language model for prediction

    Returns:
        Dictionary with primary_specialty, secondary_specialty, and explanation
    """
    # Create the prompt
    prompt = create_specialty_prediction_prompt()

    input_values = {
        "summary_text": summary_text,
        "specialties": ", ".join(specialty_list),
    }

    # Create dynamic schema from specialty list
    SpecialtyPredictionOutput = create_specialty_prediction_schema(specialty_list)

    # Use structured output - automatic validation
    structured_llm = llm.with_structured_output(SpecialtyPredictionOutput)
    chain = prompt | structured_llm

    try:
        response = chain.invoke(input_values)
        # Convert Pydantic model to dict
        return {
            "primary_specialty": response.primary_specialty,
            "secondary_specialty": response.secondary_specialty,
            "explanation": response.explanation,
        }
    except Exception as e:
        print(f"Error predicting specialty: {e}")
        return {
            "primary_specialty": None,
            "secondary_specialty": None,
            "explanation": f"Error: {str(e)}",
        }


def attending_specialty_decision(state: GraphState, llm) -> GraphState:
    """
    Predict the attending specialty for a given case (GraphState node function).

    Args:
        state: Current GraphState with summary_text and specialty_list
        llm: Language model for prediction

    Returns:
        Updated state with selected_attending_specialty
    """
    try:
        # Handle both dict and GraphState objects
        if isinstance(state, dict):
            summary_text = state.get("summary_text")
            specialty_list = state.get("specialty_list", [])
        else:
            summary_text = state.summary_text
            specialty_list = state.specialty_list

        response = predict_attending_specialty(
            summary_text=summary_text, specialty_list=specialty_list, llm=llm
        )

        print("========== selected_attending_specialty ============")
        print(response)
        print("====================================================")

        # Update state - handle both dict and GraphState
        if isinstance(state, dict):
            updated_state = state.copy()
            updated_state["selected_attending_specialty"] = response
        else:
            updated_state = state.copy()
            updated_state.selected_attending_specialty = response

        return updated_state

    except Exception as e:
        print(f"Error in attending_specialty_decision: {e}")

        # Handle errors for both types
        if isinstance(state, dict):
            updated_state = state.copy()
            updated_state["selected_attending_specialty"] = {
                "primary_specialty": None,
                "secondary_specialty": None,
                "explanation": f"Error: {str(e)}",
            }
        else:
            updated_state = state.copy()
            updated_state.selected_attending_specialty = {
                "primary_specialty": None,
                "secondary_specialty": None,
                "explanation": f"Error: {str(e)}",
            }

        return updated_state


def validate_specialty_prediction(
    prediction: Dict[str, Any], specialty_list: list
) -> Dict[str, Any]:
    """
    Validate a specialty prediction result.

    Args:
        prediction: Prediction result dictionary
        specialty_list: List of valid specialties

    Returns:
        Validation result dictionary
    """
    validation = {"valid": True, "errors": [], "warnings": []}

    # Check required fields
    required_fields = ["primary_specialty", "secondary_specialty", "explanation"]
    for field in required_fields:
        if field not in prediction:
            validation["errors"].append(f"Missing field: {field}")
            validation["valid"] = False

    # Validate primary specialty
    primary = prediction.get("primary_specialty")
    if primary is not None and primary not in specialty_list:
        validation["errors"].append(f"Invalid primary specialty: {primary}")
        validation["valid"] = False

    # Validate secondary specialty
    secondary = prediction.get("secondary_specialty")
    if secondary is not None and secondary not in specialty_list:
        validation["errors"].append(f"Invalid secondary specialty: {secondary}")
        validation["valid"] = False

    # Check for explanation
    explanation = prediction.get("explanation", "")
    if not explanation or len(explanation.strip()) < 10:
        validation["warnings"].append("Explanation is missing or too short")

    # Check if both specialties are the same
    if primary and secondary and primary == secondary:
        validation["warnings"].append("Primary and secondary specialties are identical")

    return validation


def get_specialty_confidence_score(prediction: Dict[str, Any]) -> float:
    """
    Calculate a confidence score for the specialty prediction.

    Args:
        prediction: Prediction result dictionary

    Returns:
        Confidence score between 0 and 1
    """
    score = 0.0

    # Base score for having a primary specialty
    if prediction.get("primary_specialty"):
        score += 0.6

    # Additional score for having an explanation
    explanation = prediction.get("explanation", "")
    if explanation and len(explanation.strip()) > 20:
        score += 0.3

    # Slight bonus for having a secondary specialty (shows nuanced thinking)
    if prediction.get("secondary_specialty"):
        score += 0.1

    return min(score, 1.0)


def format_specialty_prediction_output(prediction: Dict[str, Any]) -> str:
    """
    Format specialty prediction for display or logging.

    Args:
        prediction: Prediction result dictionary

    Returns:
        Formatted string representation
    """
    primary = prediction.get("primary_specialty", "None")
    secondary = prediction.get("secondary_specialty", "None")
    explanation = prediction.get("explanation", "No explanation provided")

    output = f"Primary Specialty: {primary}\n"
    output += f"Secondary Specialty: {secondary}\n"
    output += f"Explanation: {explanation}"

    return output


def get_specialty_statistics(predictions: list) -> Dict[str, Any]:
    """
    Calculate statistics for a list of specialty predictions.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Dictionary with prediction statistics
    """
    if not predictions:
        return {"total_predictions": 0}

    stats = {
        "total_predictions": len(predictions),
        "primary_specialty_counts": {},
        "secondary_specialty_counts": {},
        "has_secondary_count": 0,
        "has_explanation_count": 0,
        "average_confidence": 0.0,
    }

    confidence_scores = []

    for prediction in predictions:
        # Count primary specialties
        primary = prediction.get("primary_specialty")
        if primary:
            stats["primary_specialty_counts"][primary] = (
                stats["primary_specialty_counts"].get(primary, 0) + 1
            )

        # Count secondary specialties
        secondary = prediction.get("secondary_specialty")
        if secondary:
            stats["secondary_specialty_counts"][secondary] = (
                stats["secondary_specialty_counts"].get(secondary, 0) + 1
            )
            stats["has_secondary_count"] += 1

        # Count explanations
        explanation = prediction.get("explanation", "")
        if explanation and len(explanation.strip()) > 10:
            stats["has_explanation_count"] += 1

        # Calculate confidence
        confidence = get_specialty_confidence_score(prediction)
        confidence_scores.append(confidence)

    if confidence_scores:
        stats["average_confidence"] = sum(confidence_scores) / len(confidence_scores)

    return stats
