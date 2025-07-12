"""
Models module for the Emergency Medicine Triage RAG System.

This module provides factory functions and utilities for initializing
language models and embedding models used throughout the system.

Components:
- llm_factory: Factory functions for creating different LLM instances
- embeddings: Azure OpenAI embeddings initialization and management

Supported Models:
- DeepSeek-v3 (deepseek, deepseekv3)
- Azure OpenAI GPT-4o (gpt4o, gpt-4o, azure)
- Anthropic Claude 3.7 (claude, claude-3-7, anthropic)
"""

from .llm_factory import (
    initialize_deepseek,
    initialize_azure_openai,
    initialize_claude,
    create_llm,
    get_model_name_for_db,
    get_available_models,
)

from .embeddings import (
    initialize_embeddings,
    create_embeddings,
    get_default_embeddings,
    validate_embeddings_config,
    get_embeddings_info,
)

__all__ = [
    # LLM factory functions
    "initialize_deepseek",
    "initialize_azure_openai",
    "initialize_claude",
    "create_llm",
    "get_model_name_for_db",
    "get_available_models",
    # Embeddings functions
    "initialize_embeddings",
    "create_embeddings",
    "get_default_embeddings",
    "validate_embeddings_config",
    "get_embeddings_info",
]
