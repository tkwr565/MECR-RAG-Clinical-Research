"""
Pydantic schemas for structured outputs in the Emergency Medicine Triage RAG System.

This module defines the Pydantic models used for structured output validation
from LLM responses, ensuring consistent and machine-readable results.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class GuidelineRetrievalOutput(BaseModel):
    """Structured output for guideline retrieval decision."""

    selected_sections: List[str] = Field(
        description="List of section_title(s) exactly as they appear in the guidelines, maximum 2",
        max_items=2,
    )
    reasoning: str = Field(description="Brief explanation for section selection")


class TriageStep(BaseModel):
    """Individual step in triage prediction process."""

    category: Literal["1", "2", "3", "4", "5", "Not specified"] = Field(
        description="Triage category"
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Confidence level"
    )
    reason: str = Field(description="Reasoning for this step")


class TriagePredictionOutput(BaseModel):
    """Structured output for final triage prediction."""

    step1_clinical_risk: TriageStep = Field(description="Clinical risk assessment")
    step2_guidelines: TriageStep = Field(description="Guidelines-based assessment")
    step3_realworld_factors: TriageStep = Field(
        description="Real-world factors adjustment"
    )
    final_decision: TriageStep = Field(description="Final triage decision")


class TokenUsageMetrics(BaseModel):
    """Token usage metrics for LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def create_specialty_prediction_schema(specialty_list: List[str]):
    """
    Dynamically create a Pydantic schema for specialty prediction
    using the provided specialty list as Literal types.

    Args:
        specialty_list: List of valid specialty names

    Returns:
        Pydantic BaseModel class for specialty prediction
    """
    # Create Literal type from specialty list
    specialty_literal = Literal[tuple(specialty_list)]  # type: ignore

    class SpecialtyPredictionOutput(BaseModel):
        """Dynamic schema for specialty prediction with validated specialty names."""

        primary_specialty: specialty_literal | None = Field(  # type: ignore
            description="Primary attending specialty"
        )
        secondary_specialty: specialty_literal | None = Field(  # type: ignore
            description="Secondary attending specialty (if applicable)"
        )
        explanation: str = Field(
            description="Brief explanation of specialty selection (1-2 sentences)"
        )

    return SpecialtyPredictionOutput
