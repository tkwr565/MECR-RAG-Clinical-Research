"""
Case summarization module for the Emergency Medicine Triage RAG System.

This module handles the preprocessing of case data into concise clinical summaries
optimized for vector embedding retrieval and similarity search.
"""

import json
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..data.state import GraphState


def extract_sections_for_summary(case_json_full: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant sections from the full case JSON for summarization.

    Handles both field name formats:
    - "Clinical Presentation" vs "Clinical_Presentation"
    - "Vitals and Observations" vs "Vitals_and_Observations"

    Args:
        case_json_full: Complete case JSON data

    Returns:
        Dictionary with sections relevant for summary generation
    """
    # Handle None case
    if case_json_full is None:
        return {}

    sections_for_summary = {}

    # Extract Demographics (consistent format)
    if "Demographics" in case_json_full:
        if "Age" in case_json_full["Demographics"]:
            sections_for_summary["Age"] = case_json_full.get("Demographics", {}).get(
                "Age", ""
            )
        if "Sex" in case_json_full["Demographics"]:
            sections_for_summary["Sex"] = case_json_full.get("Demographics", {}).get(
                "Sex", ""
            )

    # Extract Vitals and Observations (handle both formats)
    vitals_key = None
    if "Vitals and Observations" in case_json_full:
        vitals_key = "Vitals and Observations"
    elif "Vitals_and_Observations" in case_json_full:
        vitals_key = "Vitals_and_Observations"

    if vitals_key:
        sections_for_summary["Vitals and Observations"] = case_json_full.get(
            vitals_key, {}
        )
        print(
            f"DEBUG: Found vitals with key '{vitals_key}': {case_json_full[vitals_key]}"
        )
    else:
        print(f"DEBUG: No vitals found. Available keys: {list(case_json_full.keys())}")

    # Extract Clinical Presentation (handle both formats)
    clinical_key = None
    if "Clinical Presentation" in case_json_full:
        clinical_key = "Clinical Presentation"
    elif "Clinical_Presentation" in case_json_full:
        clinical_key = "Clinical_Presentation"

    if clinical_key:
        sections_for_summary["Clinical Presentation"] = case_json_full.get(
            clinical_key, {}
        )
        print(
            f"DEBUG: Found clinical presentation with key '{clinical_key}': {case_json_full[clinical_key]}"
        )
    else:
        print(
            f"DEBUG: No clinical presentation found. Available keys: {list(case_json_full.keys())}"
        )

    print(f"DEBUG: Final sections_for_summary: {sections_for_summary}")
    return sections_for_summary


def create_summary_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for case summarization.

    Returns:
        ChatPromptTemplate for case summarization
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a specialized medical summarization assistant for emergency triage notes. Your task is to analyze structured JSON data from triage notes and create a concise, retrievable summary highlighting only the most important clinical information.
        ## INPUT STRUCTURE
        You will receive structured JSON data containing:
        - Demographics (sex, age)
        - Clinical Presentation (chief complaint, condition on arrival)
        - Vitals and Observations

        ## OUTPUT REQUIREMENTS
        1. Generate a concise (2-3 sentence) summary focusing on:
           - Patient demographics (sex, age)
           - Key presenting complaints
           - Abnormal physical findings
           - ONLY abnormal vital signs according to strict criteria

        2. Write in a factual, medical style optimized for vector embedding retrieval.

        ## MEDICAL TERMINOLOGY GUIDELINES
        - Interpret medical abbreviations contextually 
        - Fix obvious typos/misspellings based on medical context
        - Maintain medical precision while expanding clarity
        - For Hong Kong hospital/facility abbreviations:
          - "PMH" → "Princess Margaret Hospital"
          - "KCH" → "Kwai Chung Hospital"
          - "OAH" → "Old Age Home"
          - If uncertain about any abbreviation, do NOT expand it

        ## VITAL SIGNS ABNORMALITY CRITERIA
        ONLY mention vital signs that meet these specific abnormality thresholds:

        - Blood Pressure (Adult):
          - Hypertension: SBP > 180 mmHg or DBP > 100 mmHg
          - Hypotension: SBP < 90 mmHg

        - Heart Rate (Adults):
          - Tachycardia: HR > 120/min
          - Bradycardia: HR < 50/min

        - Temperature (Adult):
          - Fever: Temp ≥ 38°C
          - Hypothermia: Temp < 35°C

        - Blood Sugar:
          - Hyperglycemia: H'stix > 16 mmol/l
          - Hypoglycemia: H'stix < 3.9 mmol/l

        - Respiratory Status (always consider):
          - Oxygen supplementation (if present)
          - SpO2 level if abnormal
          - Abnormal respiratory rate

        - Conscious Level:
          - GCS score if not 15/15
          - Pupil abnormalities (if provided)

        - Pregnancy/Menstruation: Note if pregnant or abnormal menstrual patterns

        ## EXAMPLES OF GOOD SUMMARIES:

        ### Example 1:
        Input: [
            {{
                "Demographics": {{
                    "Sex": "F",
                    "Age": "56 years"
                }},
                "Clinical Presentation": {{
                    "Chief complaint": "R/F ? IHD, tachycardia oedema\\\\nSOB , LL edema and neck swelling x 10/7 \\\\nchest discomfort - , palpitiation",
                    "Condition on Arrival": "speak in sentence"
                }},
                "Vitals and Observations": {{
                    "Vital Signs": {{
                        "GCS": "E: 4, V: 5, M: 6, Score: 15/15",
                        "BP": "1st: 144/89 mmHg",
                        "PR/AR": "99",
                        "Temp": "36.2",
                        "SpO2": "100% on room air, --",
                        "RR": "--",
                        "PFR": "--",
                        "Limbs": "--",
                        "Pupil": "--"
                    }},
                    "Triage Intervention": {{
                        "H'stix": "",
                        "H'cue": ""
                    }},
                    "Menstruation / Pregnancy Status": "unspecified"
                }}
            }}
        ]
        
        Output: "56-year-old female presenting with shortness of breath, lower limb edema, and neck swelling for 10 days, accompanied by chest discomfort and palpitations. Patient can speak in full sentences and has been referred for evaluation of possible ischemic heart disease with tachycardia and edema. All vital signs are within normal ranges."

        ### Example 2:
        Input: [
            {{
                "Demographics": {{
                    "Sex": "M",
                    "Age": "64 years"
                }},
                "Clinical Presentation": {{
                    "Chief complaint": "SOB this morning 0730\\\\nroom air 70%",
                    "Condition on Arrival": "no lower limb oedema\\\\nsputum sound+/-"
                }},    
                "Vitals and Observations": {{
                    "Vital Signs": {{
                        "GCS": "",
                        "BP": "1st: 165/97 mmHg",
                        "PR/AR": "130",
                        "Temp": "38.1",
                        "SpO2": "96% on 6 L/min via Nasal Cannula",
                        "RR": "20",
                        "PFR": "--",
                        "Limbs": "--",
                        "Pupil": "--"
                    }},
                    "Triage Intervention": {{
                        "H'stix": "",
                        "H'cue": ""
                    }}
                }}
            }}
        ]
        Output: "64-year-old male presenting with sudden onset shortness of breath at 7:30 AM with significant hypoxia (70% on room air). Patient has tachycardia (HR 130), fever (38.1°C), and requires oxygen supplementation (6 L/min via Nasal Cannula). Clinical examination reveals sputum sounds without lower limb edema."

        ## IMPORTANT REMINDERS:
        - Focus on clinical relevance and retrieval effectiveness
        - Maintain medical accuracy while being concise
        - Only highlight abnormal vitals based on specific thresholds
        - Fix obvious errors but maintain meaning integrity
        - Adjust abnormality thresholds for pediatric cases
        """,
            ),
            (
                "user",
                """
        Case Data (JSON):
        {case_data}

        OUTPUT (Summary):
        """,
            ),
        ]
    )


def process_case_summary(state: GraphState, llm) -> GraphState:
    """
    Process case data and generate a clinical summary for vector retrieval.

    Args:
        state: Current GraphState containing case data
        llm: Language model for summary generation

    Returns:
        Updated GraphState with summary_text added
    """
    # Debug: Print state info
    print(f"DEBUG: State type: {type(state)}")
    print(
        f"DEBUG: State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}"
    )

    # Extract sections for summary - handle both dict and GraphState
    case_json_full = (
        state.get("case_json_full") if isinstance(state, dict) else state.case_json_full
    )

    print(f"DEBUG: case_json_full type: {type(case_json_full)}")
    print(f"DEBUG: case_json_full is None: {case_json_full is None}")

    # ENHANCED DEBUG: Print the actual content
    if case_json_full is not None:
        print(
            f"DEBUG: case_json_full content keys: {list(case_json_full.keys()) if isinstance(case_json_full, dict) else 'Not a dict'}"
        )
        if isinstance(case_json_full, dict):
            print(
                f"DEBUG: Demographics in case_json_full: {'Demographics' in case_json_full}"
            )
            print(
                f"DEBUG: Clinical_Presentation in case_json_full: {'Clinical_Presentation' in case_json_full}"
            )
            print(
                f"DEBUG: Vitals_and_Observations in case_json_full: {'Vitals_and_Observations' in case_json_full}"
            )

            # Print actual content
            if "Demographics" in case_json_full:
                print(f"DEBUG: Demographics content: {case_json_full['Demographics']}")
            if "Clinical_Presentation" in case_json_full:
                print(
                    f"DEBUG: Clinical_Presentation content: {case_json_full['Clinical_Presentation']}"
                )
            if "Vitals_and_Observations" in case_json_full:
                print(
                    f"DEBUG: Vitals_and_Observations content: {case_json_full['Vitals_and_Observations']}"
                )
    else:
        print("DEBUG: case_json_full is None - this is the problem!")

    if case_json_full is None:
        print("ERROR: case_json_full is None!")
        # Return empty state to avoid crash
        if isinstance(state, dict):
            updated_state = state.copy()
            updated_state["summary_text"] = "Error: No case data available"
        else:
            updated_state = state.copy()
            updated_state.summary_text = "Error: No case data available"
        return updated_state

    sections_for_summary = extract_sections_for_summary(case_json_full)

    print(f"DEBUG: sections_for_summary extracted: {sections_for_summary}")

    # Create the summary prompt
    summary_prompt = create_summary_prompt()

    # Prepare the input for the prompt
    case_data_str = json.dumps(sections_for_summary, indent=2)
    print(f"DEBUG: case_data_str being sent to LLM: {case_data_str}")

    input_values = {"case_data": case_data_str}

    # Get the LLM response
    chain = summary_prompt | llm | StrOutputParser()
    response = chain.invoke(input_values)

    print("============ update summary_text ============")
    print(response)

    # Update state - handle both dict and GraphState
    if isinstance(state, dict):
        updated_state = state.copy()
        updated_state["summary_text"] = response
    else:
        updated_state = state.copy()
        updated_state.summary_text = response

    return updated_state


def generate_summary_standalone(case_json_full: Dict[str, Any], llm) -> str:
    """
    Generate a clinical summary from case data without using GraphState.

    Args:
        case_json_full: Complete case JSON data
        llm: Language model for summary generation

    Returns:
        Generated clinical summary string
    """
    # Extract sections for summary
    sections_for_summary = extract_sections_for_summary(case_json_full)

    # Create the summary prompt
    summary_prompt = create_summary_prompt()

    # Prepare the input for the prompt
    case_data_str = json.dumps(sections_for_summary, indent=2)
    input_values = {"case_data": case_data_str}

    # Get the LLM response
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke(input_values)


def validate_summary(summary: str) -> Dict[str, Any]:
    """
    Validate the generated summary for quality and completeness.

    Args:
        summary: Generated clinical summary

    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "warnings": [],
        "length": len(summary),
        "sentence_count": summary.count(".") + summary.count("!") + summary.count("?"),
    }

    # Check length
    if len(summary) < 50:
        validation["warnings"].append("Summary might be too short")
        validation["valid"] = False
    elif len(summary) > 500:
        validation["warnings"].append("Summary might be too long")

    # Check for key medical information
    medical_terms = ["year", "old", "male", "female", "presenting", "patient"]
    if not any(term in summary.lower() for term in medical_terms):
        validation["warnings"].append(
            "Summary might be missing key medical information"
        )
        validation["valid"] = False

    # Check sentence structure
    if validation["sentence_count"] < 1:
        validation["warnings"].append(
            "Summary should contain at least one complete sentence"
        )
        validation["valid"] = False
    elif validation["sentence_count"] > 5:
        validation["warnings"].append("Summary might be too verbose (>5 sentences)")

    return validation
