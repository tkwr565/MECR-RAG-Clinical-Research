"""
Guideline retrieval module for the Emergency Medicine Triage RAG System.

This module handles the selection and retrieval of relevant triage guideline sections
based on case summaries and clinical presentations.
"""

from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..data.state import GraphState
from ..models.schemas import GuidelineRetrievalOutput
from ..utils.parsers import GuidelineSectionParser


def create_guideline_selection_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for guideline section selection.

    Note: This prompt is optimized for structured output and does not include
    JSON format instructions (handled by Pydantic schema).

    Returns:
        ChatPromptTemplate for guideline selection
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an emergency triage assistant. Your task is to select ONLY the 1-2 most critical guideline sections based on the chief complaint and abnormal vital signs.

        CRITICAL SELECTION RULES:
        1. Focus ONLY on the PRIMARY presenting complaint and documented ABNORMAL vital signs
        2. Select MAXIMUM 2 sections - prioritize life-threatening conditions first
        3. Do NOT select sections for normal findings or assumptions
        4. Consider abnormal vitals ONLY if they meet these thresholds:
           - BP: SBP > 180 or < 90 mmHg, DBP > 100 mmHg
           - HR: > 120 or < 50 (adults)
           - Temp: ≥ 38°C or < 35°C
           - Blood sugar: > 16 or < 3.9 mmol/l
           - SpO2: < 95% or requiring oxygen support
           - GCS: < 15

        SELECTION PRIORITY:
        1. Life-threatening symptoms (chest pain, SOB, altered consciousness)
        2. Abnormal vital signs meeting thresholds above
        3. Specific documented complaints
        4. If no specific sections apply, select none.
        """,
            ),
            (
                "user",
                """
        Clinical Summary: {summary_text}

        Available Guideline Sections:
        {guideline_section_metadata}

        Based on the clinical summary, which guideline sections are most relevant?
        """,
            ),
        ]
    )


def format_guideline_metadata(guideline_section_metadata: List[Dict[str, str]]) -> str:
    """
    Format guideline section metadata for the prompt.

    Args:
        guideline_section_metadata: List of guideline section metadata

    Returns:
        Formatted string for the prompt
    """
    return "\n".join(
        [
            f"Section: {meta['section_title']}\nSummary: {meta['summary']}\n"
            for meta in guideline_section_metadata
        ]
    )


def select_guideline_sections(
    summary_text: str,
    guideline_section_metadata: List[Dict[str, str]],
    llm,
    max_sections: int = 2,
) -> tuple[List[str], str]:
    """
    Select relevant guideline sections based on case summary using structured output.

    Args:
        summary_text: Clinical case summary
        guideline_section_metadata: Available guideline sections with metadata
        llm: Language model for selection
        max_sections: Maximum number of sections to select (default: 2)

    Returns:
        Tuple of (selected section titles, reasoning)
    """
    # Create the prompt
    prompt = create_guideline_selection_prompt()

    # Prepare input
    input_values = {
        "summary_text": summary_text,
        "guideline_section_metadata": format_guideline_metadata(
            guideline_section_metadata
        ),
    }

    # Use structured output - no manual parsing needed
    structured_llm = llm.with_structured_output(GuidelineRetrievalOutput)
    chain = prompt | structured_llm

    # Get structured response automatically
    response = chain.invoke(input_values)

    print("============= Native Structured Guideline Selection =============")
    print(f"Selected sections: {response.selected_sections}")
    print(f"Reasoning: {response.reasoning}")

    return response.selected_sections, response.reasoning


def retrieve_guideline_content(
    selected_sections: List[str], full_guidelines: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Retrieve the full content of selected guideline sections.

    Args:
        selected_sections: List of selected section titles
        full_guidelines: Complete guideline data with content

    Returns:
        List of guideline sections with full content
    """
    retrieved_content = []

    for section in full_guidelines:
        if section["section_title"] in selected_sections:
            retrieved_content.append(section)

    return retrieved_content


def optimized_guideline_retrieval_decision(
    state: GraphState, llm, max_sections: int = 2
) -> GraphState:
    """
    Optimized guideline retrieval with structured output focusing on chief complaint
    and abnormal vitals. Limits to 1-2 most relevant sections (GraphState node function).

    Args:
        state: Current GraphState with summary_text and guideline_section_metadata
        llm: Language model for selection
        max_sections: Maximum number of sections to select

    Returns:
        Updated state with selected_guideline_sections and guideline_reasoning
    """
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        summary_text = state.get("summary_text")
        guideline_section_metadata = state.get("guideline_section_metadata", [])
    else:
        summary_text = state.summary_text
        guideline_section_metadata = state.guideline_section_metadata

    # Get structured response with both sections and reasoning
    selected_sections, reasoning = select_guideline_sections(
        summary_text=summary_text,
        guideline_section_metadata=guideline_section_metadata,
        llm=llm,
        max_sections=max_sections,
    )

    # Update state - handle both dict and GraphState
    if isinstance(state, dict):
        updated_state = state.copy()
        updated_state["selected_guideline_sections"] = selected_sections
        updated_state["guideline_reasoning"] = reasoning
    else:
        updated_state = state.copy()
        updated_state.selected_guideline_sections = selected_sections
        updated_state.guideline_reasoning = reasoning

    return updated_state


def retrieved_guideline_section_content(
    state: GraphState, full_guidelines: List[Dict[str, str]]
) -> GraphState:
    """
    Retrieve the full content of the selected sections (GraphState node function).

    Args:
        state: Current state with selected section titles
        full_guidelines: Complete guideline data

    Returns:
        Updated state with retrieved section content
    """
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        selected_guideline_sections = state.get("selected_guideline_sections", [])
    else:
        selected_guideline_sections = state.selected_guideline_sections

    if not selected_guideline_sections:
        retrieved_content = []
    else:
        retrieved_content = retrieve_guideline_content(
            selected_sections=selected_guideline_sections,
            full_guidelines=full_guidelines,
        )

    print("============= update retrieved_guideline_content =============")
    print(retrieved_content)

    # Update state - handle both dict and GraphState
    if isinstance(state, dict):
        updated_state = state.copy()
        updated_state["retrieved_guideline_content"] = retrieved_content
    else:
        updated_state = state.copy()
        updated_state.retrieved_guideline_content = retrieved_content

    return updated_state


def validate_guideline_selection(
    selected_sections: List[str], available_sections: List[str], max_sections: int = 2
) -> Dict[str, Any]:
    """
    Validate guideline section selection.

    Args:
        selected_sections: List of selected section titles
        available_sections: List of available section titles
        max_sections: Maximum allowed sections

    Returns:
        Validation result dictionary
    """
    validation = {"valid": True, "errors": [], "warnings": []}

    # Check section count
    if len(selected_sections) > max_sections:
        validation["errors"].append(
            f"Too many sections selected: {len(selected_sections)} > {max_sections}"
        )
        validation["valid"] = False

    # Check if sections exist
    for section in selected_sections:
        if section not in available_sections:
            validation["errors"].append(f"Section not found: {section}")
            validation["valid"] = False

    # Check for duplicates
    if len(selected_sections) != len(set(selected_sections)):
        validation["warnings"].append("Duplicate sections selected")

    return validation


def format_guideline_content_for_prompt(retrieved_content: List[Dict[str, str]]) -> str:
    """
    Format retrieved guideline content for use in prompts.

    Args:
        retrieved_content: List of guideline sections with content

    Returns:
        Formatted string for prompt inclusion
    """
    if not retrieved_content:
        return "No specific guidelines retrieved."

    formatted_content = ""
    for section in retrieved_content:
        formatted_content += f"GUIDELINE: {section['section_title']}\n"
        formatted_content += f"{section['content']}...\n\n"

    return formatted_content


def get_guideline_selection_statistics(
    selections: List[List[str]], available_sections: List[str]
) -> Dict[str, Any]:
    """
    Calculate statistics for guideline selections.

    Args:
        selections: List of selection results (each is a list of selected sections)
        available_sections: List of all available section titles

    Returns:
        Dictionary with selection statistics
    """
    if not selections:
        return {"total_selections": 0}

    stats = {
        "total_selections": len(selections),
        "section_counts": {},
        "average_sections_per_case": 0.0,
        "empty_selections": 0,
        "max_sections_reached": 0,
    }

    total_sections = 0

    for selection in selections:
        # Count sections
        total_sections += len(selection)

        if not selection:
            stats["empty_selections"] += 1

        if len(selection) >= 2:  # Assuming max is 2
            stats["max_sections_reached"] += 1

        # Count individual section usage
        for section in selection:
            stats["section_counts"][section] = (
                stats["section_counts"].get(section, 0) + 1
            )

    if selections:
        stats["average_sections_per_case"] = total_sections / len(selections)

    # Calculate section popularity
    stats["most_popular_sections"] = sorted(
        stats["section_counts"].items(), key=lambda x: x[1], reverse=True
    )[:5]

    return stats


def search_guidelines_by_keywords(
    guidelines: List[Dict[str, str]],
    keywords: List[str],
    search_fields: List[str] = None,
) -> List[Dict[str, str]]:
    """
    Search guidelines by keywords in specified fields.

    Args:
        guidelines: List of guideline sections
        keywords: List of keywords to search for
        search_fields: Fields to search in (default: ['section_title', 'summary', 'content'])

    Returns:
        List of matching guideline sections
    """
    if search_fields is None:
        search_fields = ["section_title", "summary", "content"]

    matching_sections = []
    keywords_lower = [kw.lower() for kw in keywords]

    for section in guidelines:
        match_found = False

        for field in search_fields:
            if field in section:
                field_text = section[field].lower()
                if any(keyword in field_text for keyword in keywords_lower):
                    match_found = True
                    break

        if match_found:
            matching_sections.append(section)

    return matching_sections
