"""
Specialty mapping configurations for the Emergency Medicine Triage RAG System.

This module contains all specialty-related mappings, standardized lists,
and utility functions for handling medical specialty classifications.
"""

from typing import List, Dict, Optional

# Standardized specialty list used by the agent
AGENT_SPECIALTY_LIST = [
    "Medicine",
    "Surgery",
    "Obstetrics and Gynaecology",
    "Oncology",
    "Neurosurgery",
    "Orthopaedics",
    "Paediatrics",
    "Eye",
    "Psychiatry",
    "ENT",
    "Dermatology",
    "Dental",
    "Others",
]

# Mapping from standardized names to the actual values in the dataset
SPECIALTY_MAPPING = {
    "Medicine": ["Medicine", "IDU"],
    "Surgery": ["Surg", "Surgery", "Burn", "Others: UROLOGY", "Others: URO", "CTS"],
    "Obstetrics and Gynaecology": ["Gynae", "Obs/Mat"],
    "Oncology": [
        "Others: Onco",
        "Others: Oncology",
        "Others: onco",
        "Others: oncology",
        "RT",
        "Others: ONCO",
        "Others: ONC",
    ],
    "Neurosurgery": ["NS"],
    "Orthopaedics": ["O & T", "Orthopaedics"],
    "Paediatrics": ["Paed"],
    "Eye": ["Eye"],
    "Psychiatry": ["Psych"],
    "ENT": ["ENT"],
    "Dermatology": ["Derma"],
    "Dental": ["Dental"],
    "Others": ["Others"],
}

# Reverse mapping for lookup (actual values -> standardized names)
REVERSE_SPECIALTY_MAPPING = {}
for standard, variations in SPECIALTY_MAPPING.items():
    for variation in variations:
        REVERSE_SPECIALTY_MAPPING[variation] = standard


def map_specialty_to_dataset_values(specialty: str) -> List[str]:
    """
    Map a standardized specialty name to all its variations in the dataset.

    Args:
        specialty: Standardized specialty name from AGENT_SPECIALTY_LIST

    Returns:
        List of corresponding values in the dataset
    """
    if not specialty:
        return []

    return SPECIALTY_MAPPING.get(specialty, [])


def get_standardized_specialty(dataset_value: str) -> str:
    """
    Map a value from the dataset to its standardized specialty name.

    Args:
        dataset_value: Value from the dataset (e.g., "Surg", "Medicine", "IDU")

    Returns:
        Standardized specialty name from AGENT_SPECIALTY_LIST
    """
    return REVERSE_SPECIALTY_MAPPING.get(dataset_value, "Others")


def validate_specialty(specialty: str) -> bool:
    """
    Validate that a specialty is in the standardized list.

    Args:
        specialty: Specialty name to validate

    Returns:
        True if specialty is valid, False otherwise
    """
    return specialty in AGENT_SPECIALTY_LIST


def get_all_dataset_values() -> List[str]:
    """
    Get all possible dataset values across all specialties.

    Returns:
        List of all dataset values
    """
    all_values = []
    for variations in SPECIALTY_MAPPING.values():
        all_values.extend(variations)
    return all_values


def get_specialty_info() -> Dict[str, Dict]:
    """
    Get comprehensive information about all specialties.

    Returns:
        Dictionary with specialty information including counts and mappings
    """
    return {
        "standardized_specialties": AGENT_SPECIALTY_LIST,
        "total_standardized_count": len(AGENT_SPECIALTY_LIST),
        "total_dataset_values": len(get_all_dataset_values()),
        "mapping": SPECIALTY_MAPPING,
        "reverse_mapping": REVERSE_SPECIALTY_MAPPING,
    }
