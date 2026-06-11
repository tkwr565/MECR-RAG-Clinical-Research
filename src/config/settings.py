"""
Configuration settings for the Emergency Medicine Triage RAG System.

This module handles environment variable loading and validation,
as well as defining system-wide constants and paths.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Central configuration class for the triage system."""

    def __init__(self):
        self.validate_environment()

    # Required environment variables
    REQUIRED_ENV_VARS = [
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "AZURE_OPENAI_LLM_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL_NAME",
    ]

    # Azure OpenAI Configuration
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME: str = os.getenv(
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
    )
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_LLM_DEPLOYMENT_NAME: str = os.getenv(
        "AZURE_OPENAI_LLM_DEPLOYMENT_NAME"
    )
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")

    # DeepSeek Configuration
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL_NAME: str = "deepseek-v4-pro"

    # Anthropic Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME: str = os.getenv("ANTHROPIC_MODEL_NAME")

    # Database Paths (relative to project root)
    DB_BASE_DIR: str = "db"
    TRIAGE_GUIDELINES_PATH: str = "db/triage_sections_with_summaries_deepseek-v4-pro.json"
    PAST_CASE_BASE_DIR: str = "db/past_case"

    # Model Configuration
    DEFAULT_TEMPERATURE: float = 0
    DEFAULT_MAX_TOKENS: int = 8192
    DEFAULT_STREAMING: bool = True

    # Retrieval Configuration
    K_CASES: int = 5  # Number of past cases to retrieve
    METADATA_THRESHOLD: float = 1.0
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_GUIDELINE_SECTIONS: int = 2

    # Output Configuration
    DEFAULT_OUTPUT_DIR: str = "outputs/prediction_results"

    @classmethod
    def validate_environment(cls) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

    @classmethod
    def get_vector_db_path(cls, model: str) -> str:
        """Get the vector database path for a specific model."""
        return f"{cls.PAST_CASE_BASE_DIR}/case_vectordb_{model}_3000case"

    @classmethod
    def get_case_json_dir(cls, model: str) -> str:
        """Get the case JSON directory path for a specific model."""
        return f"{cls.get_vector_db_path(model)}/json_store"

    @classmethod
    def get_summary_vectordb_path(cls, model: str) -> str:
        """Get the summary vector database path for a specific model."""
        return f"{cls.get_vector_db_path(model)}/summary_vectordb"


# Create global settings instance
settings = Settings()

# Model name mappings for vector database paths
MODEL_NAMES = {"deepseek": "deepseek-v4-pro", "gpt4o": "gpt-4o", "claude": "claude-3-7"}


def get_model_db_name(model_key: str) -> str:
    """Get the database model name from a model key."""
    return MODEL_NAMES.get(model_key, model_key)
