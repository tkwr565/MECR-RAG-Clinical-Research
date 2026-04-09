"""
State management for the Emergency Medicine Triage RAG System.

This module defines the GraphState class that manages data flow
through the LangGraph pipeline stages.
"""

from typing import Dict, List, Any, Optional


class GraphState(dict):
    """
    State for the retrieval graph.

    This class manages the data flow through the triage assessment pipeline,
    storing intermediate results and passing data between processing nodes.

    Attributes:
        summary_text: Clinical summary for vector search of past case retrieval
        metadata_filter: Filter criteria for past case retrieval
        case_json_full: Complete case JSON data
        guideline_section_metadata: Guideline section metadata (title, summary)
        selected_guideline_sections: List of selected section titles
        retrieved_guideline_content: Content of selected guideline sections
        specialty_list: Standardized specialty list
        selected_attending_specialty: Predicted specialty information
        retrieved_cases: Retrieved similar past cases with metadata
        retrieved_cases_content: Full content of retrieved past cases
        past_cases_context: Formatted context from past cases
        final_assessment: Final triage assessment and reasoning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize with default None values for type hints
        self.setdefault("summary_text", None)
        self.setdefault("metadata_filter", {})
        self.setdefault("case_json_full", {})
        self.setdefault("guideline_section_metadata", [])
        self.setdefault("selected_guideline_sections", None)
        self.setdefault("guideline_reasoning", None)  # New: reasoning for guideline selection
        self.setdefault("retrieved_guideline_content", None)
        self.setdefault("specialty_list", [])
        self.setdefault("selected_attending_specialty", None)
        self.setdefault("retrieved_cases", [])
        self.setdefault("retrieved_cases_content", [])
        self.setdefault("past_cases_context", "")
        self.setdefault("final_assessment", None)
        self.setdefault("final_category", None)  # New: extracted final category
        self.setdefault("final_confidence", None)  # New: extracted final confidence
        # Fields for error handling
        self.setdefault("processing_failed", False)
        self.setdefault("failure_reason", "")
        self.setdefault("failed_node", "")

    # Type-safe property accessors
    @property
    def summary_text(self) -> Optional[str]:
        """Embedding text for vector search of past case retrieval."""
        return self.get("summary_text")

    @summary_text.setter
    def summary_text(self, value: Optional[str]):
        self["summary_text"] = value

    @property
    def metadata_filter(self) -> Dict[str, Any]:
        """Filter for past case retrieval."""
        return self.get("metadata_filter", {})

    @metadata_filter.setter
    def metadata_filter(self, value: Dict[str, Any]):
        self["metadata_filter"] = value

    @property
    def case_json_full(self) -> Dict[str, Any]:
        """Full case JSON data."""
        return self.get("case_json_full", {})

    @case_json_full.setter
    def case_json_full(self, value: Dict[str, Any]):
        self["case_json_full"] = value

    @property
    def guideline_section_metadata(self) -> List[Dict[str, str]]:
        """Guideline section metadata: title, summary."""
        return self.get("guideline_section_metadata", [])

    @guideline_section_metadata.setter
    def guideline_section_metadata(self, value: List[Dict[str, str]]):
        self["guideline_section_metadata"] = value

    @property
    def selected_guideline_sections(self) -> Optional[List[str]]:
        """Selected section titles."""
        return self.get("selected_guideline_sections")

    @selected_guideline_sections.setter
    def selected_guideline_sections(self, value: Optional[List[str]]):
        self["selected_guideline_sections"] = value

    @property
    def retrieved_guideline_content(self) -> Optional[List[Dict[str, str]]]:
        """Selected sections content."""
        return self.get("retrieved_guideline_content")

    @retrieved_guideline_content.setter
    def retrieved_guideline_content(self, value: Optional[List[Dict[str, str]]]):
        self["retrieved_guideline_content"] = value

    @property
    def specialty_list(self) -> List[str]:
        """Standardized specialty list."""
        return self.get("specialty_list", [])

    @specialty_list.setter
    def specialty_list(self, value: List[str]):
        self["specialty_list"] = value

    @property
    def selected_attending_specialty(self) -> Optional[Dict[str, Any]]:
        """Selected specialty."""
        return self.get("selected_attending_specialty")

    @selected_attending_specialty.setter
    def selected_attending_specialty(self, value: Optional[Dict[str, Any]]):
        self["selected_attending_specialty"] = value

    @property
    def retrieved_cases(self) -> List[Dict[str, Any]]:
        """Retrieved cases with metadata."""
        return self.get("retrieved_cases", [])

    @retrieved_cases.setter
    def retrieved_cases(self, value: List[Dict[str, Any]]):
        self["retrieved_cases"] = value

    @property
    def retrieved_cases_content(self) -> List[Dict[str, Any]]:
        """Full content of retrieved cases."""
        return self.get("retrieved_cases_content", [])

    @retrieved_cases_content.setter
    def retrieved_cases_content(self, value: List[Dict[str, Any]]):
        self["retrieved_cases_content"] = value

    @property
    def past_cases_context(self) -> str:
        """Formatted context from past cases."""
        return self.get("past_cases_context", "")

    @past_cases_context.setter
    def past_cases_context(self, value: str):
        self["past_cases_context"] = value

    @property
    def final_assessment(self) -> Optional[Dict[str, Any]]:
        """Final triage assessment (structured output dict or legacy string)."""
        return self.get("final_assessment")

    @final_assessment.setter
    def final_assessment(self, value: Optional[Dict[str, Any]]):
        self["final_assessment"] = value

    @property
    def guideline_reasoning(self) -> Optional[str]:
        """Reasoning for guideline section selection."""
        return self.get("guideline_reasoning")

    @guideline_reasoning.setter
    def guideline_reasoning(self, value: Optional[str]):
        self["guideline_reasoning"] = value

    @property
    def final_category(self) -> Optional[str]:
        """Extracted final triage category."""
        return self.get("final_category")

    @final_category.setter
    def final_category(self, value: Optional[str]):
        self["final_category"] = value

    @property
    def final_confidence(self) -> Optional[str]:
        """Extracted final confidence level."""
        return self.get("final_confidence")

    @final_confidence.setter
    def final_confidence(self, value: Optional[str]):
        self["final_confidence"] = value

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state.

        Returns:
            Dictionary with key state information
        """
        return {
            "has_summary_text": self.summary_text is not None,
            "has_case_data": bool(self.case_json_full),
            "guideline_sections_count": len(self.guideline_section_metadata),
            "selected_sections_count": (
                len(self.selected_guideline_sections)
                if self.selected_guideline_sections
                else 0
            ),
            "retrieved_guidelines_count": (
                len(self.retrieved_guideline_content)
                if self.retrieved_guideline_content
                else 0
            ),
            "specialty_predicted": self.selected_attending_specialty is not None,
            "retrieved_cases_count": len(self.retrieved_cases),
            "has_final_assessment": self.final_assessment is not None,
        }

    def validate(self) -> List[str]:
        """
        Validate the state for common issues.

        Returns:
            List of validation error messages
        """
        errors = []

        if not self.case_json_full:
            errors.append("Missing case JSON data")

        if not self.guideline_section_metadata:
            errors.append("Missing guideline section metadata")

        if not self.specialty_list:
            errors.append("Missing specialty list")

        return errors
