"""
Triage assessment module for the Emergency Medicine Triage RAG System.

This module handles the final triage category prediction and reasoning
using integrated information from guidelines and past cases.
"""

from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..data.state import GraphState
from ..retrieval.guideline_retriever import format_guideline_content_for_prompt
from ..retrieval.case_retriever import format_past_cases_context


def create_triage_assessment_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for simplified 3-step triage assessment.

    Returns:
        ChatPromptTemplate for triage assessment
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an emergency triage specialist. Use this simplified 3-step approach to determine the appropriate triage category.

        TRIAGE CATEGORY DEFINITIONS:
        | Triage Category | Patient Conditions | Actions of Staff | Target Response Time (Time from registration to medical consultation) |
        |---------------|------------------|----------------|---------------------------------------------------------------------|
        | 1 Critical | • Suffer from a life-threatening condition(s) caused by a major event • With unstable vital signs requiring immediate resuscitation | • Direct patient to resuscitation room • Attend patient immediately by a team comprising medical and nursing staff | • Immediate • 100% of cases within the target response time |
        | 2 Emergency | • Suffer from a potentially life-threatening condition • Borderline vital signs but with potential risk of rapid deterioration • Require emergency treatment and immediate continuous close monitoring | • Direct patient to resuscitation room / treatment cubicle • Offer medical attention and immediate continuous close monitoring within 15 mins. | • < 15 mins • 95% of cases within the target response time |
        | 3 Urgent | • Suffer from a major condition with potential risk of deterioration • Stable vital signs | • Direct patient to cubicle | • < 30 mins • 90% of cases within the target response time |
        | 4 Semi-urgent | • Suffer from acute but stable condition(s) • Stable vital signs • Can afford to wait some time without serious complications | • Direct patient to cubicle / walk-in clinic | • < 120 mins |
        | 5 Non-urgent | • Suffer from minor and stable condition(s) (including acute and non-acute conditions) • Can afford to wait without deterioration • Stable vital signs | • Direct patient to walk-in clinic | Remarks: • Conditions can be treated in primary health care facilities • Should be based on clinical judgement only. Economic, social factors and availability of facilities should not be taken into consideration. |

        USE THIS EXACT 3-STEP FORMAT:

        ### STEP 1: CLINICAL RISK ASSESSMENT (Based on Triage Definitions)
        **Focus:** Determine overall CLINICAL RISK and URGENCY based on triage category definitions
        **Patient Profile:** [Age, sex, key symptoms, abnormal vitals only]
        **Risk of Deterioration:** [Immediate/High/Moderate/Low - based on clinical instability]
        **Urgency Level:** [Life-threatening/Potentially life-threatening/Major/Acute/Minor]
        **Initial Triage Category (Clinical Risk):** [1/2/3/4/5]

        ### STEP 2: GUIDELINE-BASED ASSESSMENT (Medical Condition Specific)
        **Focus:** Apply specific medical guidelines for documented conditions
        **Most Relevant Guideline:** [Title of guideline, or "None specific"]
        **Specific Medical Condition:** [Condition addressed by guideline]
        **Guideline Criteria:** [Specific criteria from guideline]
        **Criteria Met:** [Yes/No with evidence]
        **Guideline-Recommended Category:** [1/2/3/4/5, or "Not specified"]

        ### STEP 3: REAL-WORLD FACTORS ANALYSIS (Past Case Patterns)
        **Focus:** Analyze factors OUTSIDE formal guidelines that influence real-world triage decisions
        
        **Case-by-Case Analysis:**
        #### Past Case 1:
        - **Demographics & Status:** [Age, ambulatory status, institutionalized status, referral status]
        - **Co-morbidities:** [Relevant past health conditions]
        - **Assigned Category:** [Category]
        - **Key Similarities to Current Case:** [List similarities]
        - **Key Differences from Current Case:** [List differences]
        - **Non-Guideline Factors:** [Age-related, mobility, social factors that may have influenced decision]

        [Repeat for each case]

        #### Pattern Recognition:
        - **Common Factors in Similar Cases:** [Factors appearing across multiple cases]
        - **Deviation Patterns from Guidelines:** [How similar cases deviate from strict guideline application]
        - **Implicit Decision Rules:** [Unwritten rules evident in past case decisions]
        - **Impact of Special Factors:** [How factors like age, ambulatory status, etc. affect categorization]

        ### FINAL DECISION
        **Step 1 Category (Clinical Risk):** [1/2/3/4/5]
        **Step 2 Category (Guidelines):** [1/2/3/4/5 or "Not specified"]
        **Step 3 Adjustment (Real-World Factors):** [Higher/Lower/No change]
        **FINAL TRIAGE CATEGORY:** [1/2/3/4/5]
        **CONFIDENCE:** [High/Medium/Low]
        **KEY RATIONALE:** [Explain which step(s) drove the final decision and any adjustments made for real-world factors]

        CRITICAL INSTRUCTIONS:
        - Step 1: Focus purely on clinical urgency and risk of deterioration per triage definitions
        - Step 2: Apply medical guidelines for specific documented conditions only
        - Step 3: Analyze ALL provided past cases individually - look for demographic, social, and contextual factors that influenced real-world decisions beyond clinical guidelines
        - Consider non-guideline factors: age extremes, ambulatory status, institutionalized patients, referral patterns, complex co-morbidities
        - If past cases show different categories than Steps 1&2 suggest, identify what real-world factors caused the deviation
        """,
            ),
            (
                "user",
                """
        Current Case:
        {case_summary}
        
        Relevant Guidelines:
        {guideline_content}
        
        Similar Past Cases:
        {past_cases_context}
        
        Please provide your 3-step triage assessment:
        """,
            ),
        ]
    )


def prepare_case_summary_for_assessment(state: GraphState) -> str:
    """
    Prepare case summary section for triage assessment.

    Args:
        state: Current GraphState with case data

    Returns:
        Formatted case summary string
    """
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        summary_text = state.get("summary_text", "")
        case_json_full = state.get("case_json_full", {})
    else:
        summary_text = state.summary_text or ""
        case_json_full = state.case_json_full or {}

    case_summary = f"""
    Clinical Summary: {summary_text}
    
    Demographics: {case_json_full.get('Demographics', {})}
    """
    return case_summary


def generate_triage_assessment(
    case_summary: str, guideline_content: str, past_cases_context: str, llm
) -> str:
    """
    Generate a comprehensive triage assessment using the 3-step approach.

    Args:
        case_summary: Formatted case summary
        guideline_content: Formatted guideline content
        past_cases_context: Formatted past cases context
        llm: Language model for assessment generation

    Returns:
        Complete triage assessment string
    """
    # Create the prompt
    prompt = create_triage_assessment_prompt()

    # Prepare input
    input_values = {
        "case_summary": case_summary,
        "guideline_content": guideline_content,
        "past_cases_context": past_cases_context,
    }

    # Get LLM response
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(input_values)

    return response


def simplified_triage_prediction(state: GraphState, llm) -> GraphState:
    """
    Simplified 3-step triage prediction focusing on key decision factors (GraphState node function).

    Args:
        state: Current GraphState with all retrieved information
        llm: Language model for assessment generation

    Returns:
        Updated state with final_assessment
    """
    # Handle both dict and GraphState objects
    if isinstance(state, dict):
        retrieved_guideline_content = state.get("retrieved_guideline_content", [])
        retrieved_cases_content = state.get("retrieved_cases_content", [])
    else:
        retrieved_guideline_content = state.retrieved_guideline_content or []
        retrieved_cases_content = state.retrieved_cases_content or []

    # Prepare case summary
    case_summary = prepare_case_summary_for_assessment(state)

    # Prepare guideline content
    guideline_content = format_guideline_content_for_prompt(retrieved_guideline_content)

    # Prepare past cases context
    past_cases_context = format_past_cases_context(retrieved_cases_content)

    # Generate assessment
    assessment = generate_triage_assessment(
        case_summary=case_summary,
        guideline_content=guideline_content,
        past_cases_context=past_cases_context,
        llm=llm,
    )

    print("======== SIMPLIFIED TRIAGE ASSESSMENT ========")
    print(assessment)
    print("=" * 60)

    # Update state - handle both dict and GraphState
    if isinstance(state, dict):
        updated_state = state.copy()
        updated_state["final_assessment"] = assessment
    else:
        updated_state = state.copy()
        updated_state.final_assessment = assessment

    return updated_state


def extract_final_category_from_assessment(assessment: str) -> Dict[str, Any]:
    """
    Extract the final triage category and confidence from the assessment text.

    Args:
        assessment: Complete triage assessment text

    Returns:
        Dictionary with extracted category, confidence, and rationale
    """
    import re

    result = {"category": None, "confidence": None, "rationale": None}

    # Extract final triage category with multiple robust patterns
    final_category_patterns = [
        r"\*\*FINAL TRIAGE CATEGORY:\*\*\s*\*\*Category\s*(\d)(?:[^\d]*)\*\*",  # **FINAL TRIAGE CATEGORY:** **Category 3 (Urgent)**
        r"\*\*FINAL TRIAGE CATEGORY:\*\*\s*\*\*(\d)(?:[^\d]*)\*\*",  # **FINAL TRIAGE CATEGORY:** **3 (Urgent)**
        r"\*\*FINAL TRIAGE CATEGORY:\s*Category\s*(\d)(?:[^\d]*)\*\*",  # **FINAL TRIAGE CATEGORY: Category 3 (Urgent)**
        r"\*\*FINAL TRIAGE CATEGORY:\s*(\d)(?:[^\d]*)\*\*",  # **FINAL TRIAGE CATEGORY: 3 (Urgent)**
        r"\*\*FINAL TRIAGE CATEGORY:\*\*\s*\[(\d)\]",  # **FINAL TRIAGE CATEGORY:** [4]
        r"\*\*FINAL TRIAGE CATEGORY:\*\*\s*Category\s*(\d)",  # **FINAL TRIAGE CATEGORY:** Category 3
        r"\*\*FINAL TRIAGE CATEGORY:\*\*\s*(\d)",  # **FINAL TRIAGE CATEGORY:** 3
        r"FINAL TRIAGE CATEGORY:\s*\[(\d)\]",  # FINAL TRIAGE CATEGORY: [4]
        r"FINAL TRIAGE CATEGORY:\s*Category\s*(\d)",  # FINAL TRIAGE CATEGORY: Category 3
        r"FINAL TRIAGE CATEGORY:\s*(\d)",  # FINAL TRIAGE CATEGORY: 3
        r"\*\*Final Triage Category:\*\*\s*\*\*Category\s*(\d)(?:[^\d]*)\*\*",  # **Final Triage Category:** **Category 3 (Urgent)**
        r"\*\*Final Triage Category:\*\*\s*\*\*(\d)(?:[^\d]*)\*\*",  # **Final Triage Category:** **3 (Urgent)**
        r"\*\*Final Triage Category:\s*Category\s*(\d)(?:[^\d]*)\*\*",  # **Final Triage Category: Category 3 (Urgent)**
        r"\*\*Final Triage Category:\s*(\d)(?:[^\d]*)\*\*",  # **Final Triage Category: 3 (Urgent)**
        r"\*\*Final Triage Category:\*\*\s*\[(\d)\]",  # **Final Triage Category:** [4]
        r"\*\*Final Triage Category:\*\*\s*Category\s*(\d)",  # **Final Triage Category:** Category 3
        r"\*\*Final Triage Category:\*\*\s*(\d)",  # **Final Triage Category:** 3
        r"Final Triage Category:\s*\[(\d)\]",  # Final Triage Category: [4]
        r"Final Triage Category:\s*Category\s*(\d)",  # Final Triage Category: Category 3
        r"Final Triage Category:\s*(\d)",  # Final Triage Category: 3
    ]

    for pattern in final_category_patterns:
        match = re.search(pattern, assessment, re.IGNORECASE)
        if match:
            try:
                result["category"] = int(match.group(1))
                print(
                    f"DEBUG: Found category {result['category']} with pattern: {pattern}"
                )
                break
            except ValueError:
                continue

    # Extract confidence with multiple patterns
    confidence_patterns = [
        r"\*\*CONFIDENCE:\*\*\s*\*\*(High|Medium|Low)\*\*",  # **CONFIDENCE:** **High**
        r"\*\*CONFIDENCE:\*\*\s*(High|Medium|Low)",  # **CONFIDENCE:** High
        r"\*\*Confidence:\*\*\s*\*\*(High|Medium|Low)\*\*",  # **Confidence:** **High**
        r"\*\*Confidence:\*\*\s*(High|Medium|Low)",  # **Confidence:** High
        r"CONFIDENCE:\s*\*\*(High|Medium|Low)\*\*",  # CONFIDENCE: **High**
        r"CONFIDENCE:\s*(High|Medium|Low)",  # CONFIDENCE: High
        r"Confidence:\s*\*\*(High|Medium|Low)\*\*",  # Confidence: **High**
        r"Confidence:\s*(High|Medium|Low)",  # Confidence: High
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, assessment, re.IGNORECASE)
        if match:
            result["confidence"] = match.group(1).lower()
            print(
                f"DEBUG: Found confidence {result['confidence']} with pattern: {pattern}"
            )
            break

    # Extract key rationale with multiple patterns
    rationale_patterns = [
        r"\*\*KEY RATIONALE:\*\*\s*\*\*(.+?)\*\*(?:\n|\Z)",  # **KEY RATIONALE:** **text**
        r"\*\*KEY RATIONALE:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)",  # **KEY RATIONALE:** text (until double newline or next bold)
        r"\*\*Key Rationale:\*\*\s*\*\*(.+?)\*\*(?:\n|\Z)",  # **Key Rationale:** **text**
        r"\*\*Key Rationale:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)",  # **Key Rationale:** text
        r"KEY RATIONALE:\s*\*\*(.+?)\*\*(?:\n|\Z)",  # KEY RATIONALE: **text**
        r"KEY RATIONALE:\s*(.+?)(?:\n\n|\n\*\*|\Z)",  # KEY RATIONALE: text
        r"Key Rationale:\s*\*\*(.+?)\*\*(?:\n|\Z)",  # Key Rationale: **text**
        r"Key Rationale:\s*(.+?)(?:\n\n|\n\*\*|\Z)",  # Key Rationale: text
    ]

    for pattern in rationale_patterns:
        match = re.search(pattern, assessment, re.IGNORECASE | re.DOTALL)
        if match:
            result["rationale"] = match.group(1).strip()
            print(f"DEBUG: Found rationale with pattern: {pattern}")
            break

    print(f"DEBUG: Final extraction result: {result}")
    return result


def validate_triage_assessment(assessment: str) -> Dict[str, Any]:
    """
    Validate the completeness and structure of a triage assessment.

    Args:
        assessment: Triage assessment text

    Returns:
        Validation result dictionary
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "completeness_score": 0.0,
    }

    required_sections = [
        "STEP 1: CLINICAL RISK ASSESSMENT",
        "STEP 2: GUIDELINE-BASED ASSESSMENT",
        "STEP 3: REAL-WORLD FACTORS ANALYSIS",
        "FINAL DECISION",
    ]

    sections_found = 0
    for section in required_sections:
        if section in assessment.upper():
            sections_found += 1
        else:
            validation["errors"].append(f"Missing section: {section}")

    validation["completeness_score"] = sections_found / len(required_sections)

    if sections_found < len(required_sections):
        validation["valid"] = False

    # Check for final category
    extracted = extract_final_category_from_assessment(assessment)
    if extracted["category"] is None:
        validation["errors"].append("No final triage category found")
        validation["valid"] = False
    elif not (1 <= extracted["category"] <= 5):
        validation["errors"].append(f"Invalid triage category: {extracted['category']}")
        validation["valid"] = False

    # Check length
    if len(assessment) < 500:
        validation["warnings"].append("Assessment might be too brief")
    elif len(assessment) > 3000:
        validation["warnings"].append("Assessment might be too verbose")

    return validation


def get_assessment_statistics(assessments: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of triage assessments.

    Args:
        assessments: List of assessment texts

    Returns:
        Dictionary with assessment statistics
    """
    if not assessments:
        return {"total_assessments": 0}

    stats = {
        "total_assessments": len(assessments),
        "category_distribution": {},
        "confidence_distribution": {},
        "avg_length": 0.0,
        "completeness_scores": [],
        "valid_assessments": 0,
    }

    total_length = 0

    for assessment in assessments:
        # Extract information
        extracted = extract_final_category_from_assessment(assessment)
        validation = validate_triage_assessment(assessment)

        # Count categories
        if extracted["category"]:
            category = extracted["category"]
            stats["category_distribution"][category] = (
                stats["category_distribution"].get(category, 0) + 1
            )

        # Count confidence levels
        if extracted["confidence"]:
            confidence = extracted["confidence"]
            stats["confidence_distribution"][confidence] = (
                stats["confidence_distribution"].get(confidence, 0) + 1
            )

        # Track validity
        if validation["valid"]:
            stats["valid_assessments"] += 1

        # Track completeness
        stats["completeness_scores"].append(validation["completeness_score"])

        # Track length
        total_length += len(assessment)

    stats["avg_length"] = total_length / len(assessments)

    if stats["completeness_scores"]:
        stats["avg_completeness"] = sum(stats["completeness_scores"]) / len(
            stats["completeness_scores"]
        )

    return stats
