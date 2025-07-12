"""
Output parsers for the Emergency Medicine Triage RAG System.

This module contains custom parsers for structured LLM outputs,
including specialty prediction and other JSON-based responses.
"""

from typing import Dict, Any, Optional
from langchain_core.output_parsers import JsonOutputParser

from ..config.specialty_mapping import AGENT_SPECIALTY_LIST


class SpecialtyPrediction(JsonOutputParser):
    """
    Custom JSON parser for specialty prediction responses.

    Validates that predicted specialties are in the allowed list
    and handles parsing errors gracefully.
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM output for specialty prediction.

        Args:
            text: Raw LLM output text

        Returns:
            Dictionary with primary_specialty, secondary_specialty, and explanation
        """
        try:
            parsed = super().parse(text)

            # Ensure the parsed response has the required fields
            if not isinstance(parsed, dict):
                return {
                    "primary_specialty": None,
                    "secondary_specialty": None,
                    "explanation": "Failed to parse response",
                }

            # Extract and validate specialties
            primary = parsed.get("primary_specialty")
            secondary = parsed.get("secondary_specialty")
            explanation = parsed.get("explanation", "")

            # Validate that the specialties are in our list
            if primary and primary not in AGENT_SPECIALTY_LIST:
                primary = None
            if secondary and secondary not in AGENT_SPECIALTY_LIST:
                secondary = None

            return {
                "primary_specialty": primary,
                "secondary_specialty": secondary,
                "explanation": explanation,
            }

        except Exception as e:
            print(f"Error parsing specialty prediction: {e}")
            return {
                "primary_specialty": None,
                "secondary_specialty": None,
                "explanation": f"Error: {str(e)}",
            }


class TriageCategoryParser(JsonOutputParser):
    """
    Custom JSON parser for triage category predictions.

    Validates that predicted categories are valid (1-5) and handles
    confidence scores and explanations.
    """

    def __init__(self):
        super().__init__()
        self.VALID_CATEGORIES = [1, 2, 3, 4, 5]

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM output for triage category prediction.

        Args:
            text: Raw LLM output text

        Returns:
            Dictionary with category, confidence, and explanation
        """
        try:
            parsed = super().parse(text)

            if not isinstance(parsed, dict):
                return {
                    "category": None,
                    "confidence": None,
                    "explanation": "Failed to parse response",
                }

            # Extract and validate category
            category = parsed.get("category")
            if category is not None:
                try:
                    category = int(category)
                    if category not in self.VALID_CATEGORIES:
                        category = None
                except (ValueError, TypeError):
                    category = None

            # Extract confidence
            confidence = parsed.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                    # Ensure confidence is between 0 and 1
                    if not (0 <= confidence <= 1):
                        confidence = None
                except (ValueError, TypeError):
                    confidence = None

            explanation = parsed.get("explanation", "")

            return {
                "category": category,
                "confidence": confidence,
                "explanation": explanation,
            }

        except Exception as e:
            print(f"Error parsing triage category: {e}")
            return {
                "category": None,
                "confidence": None,
                "explanation": f"Error: {str(e)}",
            }


class GuidelineSectionParser:
    """
    Parser for guideline section selection responses.

    Handles comma-separated lists of section titles and validates
    against available sections.
    """

    def __init__(self, available_sections: Optional[list] = None):
        """
        Initialize parser with available section titles.

        Args:
            available_sections: List of valid section titles
        """
        self.available_sections = available_sections or []

    def parse(self, text: str, max_sections: int = 2) -> list:
        """
        Parse guideline section selection response.

        Args:
            text: Raw LLM output text
            max_sections: Maximum number of sections to return

        Returns:
            List of selected section titles
        """
        text = text.strip()

        # Handle "None" response
        if "None" in text or not text:
            return []

        # Split by comma and clean
        sections = [section.strip() for section in text.split(",")]

        # Validate against available sections if provided
        if self.available_sections:
            valid_sections = []
            for section in sections:
                if section in self.available_sections:
                    valid_sections.append(section)
                else:
                    # Try fuzzy matching for common variations
                    for available in self.available_sections:
                        if (
                            section.lower() in available.lower()
                            or available.lower() in section.lower()
                        ):
                            valid_sections.append(available)
                            break
            sections = valid_sections

        # Limit to max_sections
        return sections[:max_sections]


def parse_json_safely(text: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely parse JSON text with fallback to default.

    Args:
        text: JSON text to parse
        default: Default value if parsing fails

    Returns:
        Parsed dictionary or default
    """
    if default is None:
        default = {}

    try:
        parser = JsonOutputParser()
        return parser.parse(text)
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return default


def extract_category_from_text(text: str) -> Optional[int]:
    """
    Extract triage category number from free text.

    Args:
        text: Text potentially containing category information

    Returns:
        Extracted category number or None
    """
    import re

    # Look for patterns like "Category 3", "Cat 2", "Triage 4", etc.
    patterns = [
        r"category\s*(\d)",
        r"cat\s*(\d)",
        r"triage\s*(\d)",
        r"level\s*(\d)",
        r"priority\s*(\d)",
    ]

    text_lower = text.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                category = int(matches[0])
                if 1 <= category <= 5:
                    return category
            except ValueError:
                continue

    return None


def validate_parser_output(
    output: Dict[str, Any], required_fields: list, field_types: Dict[str, type] = None
) -> Dict[str, Any]:
    """
    Validate parser output against required fields and types.

    Args:
        output: Parser output dictionary
        required_fields: List of required field names
        field_types: Dictionary mapping field names to expected types

    Returns:
        Validation result with status and errors
    """
    if field_types is None:
        field_types = {}

    validation = {
        "valid": True,
        "missing_fields": [],
        "type_errors": [],
        "warnings": [],
    }

    # Check required fields
    for field in required_fields:
        if field not in output or output[field] is None:
            validation["missing_fields"].append(field)
            validation["valid"] = False

    # Check field types
    for field, expected_type in field_types.items():
        if field in output and output[field] is not None:
            if not isinstance(output[field], expected_type):
                validation["type_errors"].append(
                    f"{field}: expected {expected_type.__name__}, got {type(output[field]).__name__}"
                )
                validation["valid"] = False

    return validation
