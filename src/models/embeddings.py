"""
Embeddings module for the Emergency Medicine Triage RAG System.

This module provides functions to initialize and manage Azure OpenAI embeddings
for vector database operations and similarity search.
"""

from typing import Optional
from langchain_openai import AzureOpenAIEmbeddings

from ..config.settings import settings


def initialize_embeddings(
    azure_deployment: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
) -> AzureOpenAIEmbeddings:
    """
    Initialize Azure OpenAI embeddings.

    Args:
        azure_deployment: Azure deployment name for embeddings (defaults to settings)
        azure_endpoint: Azure endpoint URL (defaults to settings)
        api_key: Azure API key (defaults to settings)
        api_version: Azure API version (defaults to settings)

    Returns:
        AzureOpenAIEmbeddings: Initialized embeddings client
    """
    return AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment
        or settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        azure_endpoint=azure_endpoint or settings.AZURE_OPENAI_ENDPOINT,
        api_key=api_key or settings.AZURE_OPENAI_API_KEY,
        api_version=api_version or settings.AZURE_OPENAI_API_VERSION,
    )


def initialize_standard_embeddings(
    azure_deployment: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
) -> AzureOpenAIEmbeddings:
    """
    Initialize Azure OpenAI embeddings using the standard deployment.

    Args:
        azure_deployment: Azure deployment name for embeddings (defaults to standard setting)
        azure_endpoint: Azure endpoint URL (defaults to settings)
        api_key: Azure API key (defaults to settings)
        api_version: Azure API version (defaults to settings)

    Returns:
        AzureOpenAIEmbeddings: Initialized embeddings client
    """
    return AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment
        or settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        azure_endpoint=azure_endpoint or settings.AZURE_OPENAI_ENDPOINT,
        api_key=api_key or settings.AZURE_OPENAI_API_KEY,
        api_version=api_version or settings.AZURE_OPENAI_API_VERSION,
    )


def create_embeddings(**kwargs) -> AzureOpenAIEmbeddings:
    """
    Factory function to create embeddings.

    Args:
        **kwargs: Additional arguments to pass to the initializer

    Returns:
        AzureOpenAIEmbeddings: Initialized embeddings client
    """
    return initialize_embeddings(**kwargs)


def get_default_embeddings() -> AzureOpenAIEmbeddings:
    """
    Get the default embeddings client.

    Returns:
        AzureOpenAIEmbeddings: Default embeddings client
    """
    return initialize_embeddings()


def validate_embeddings_config() -> bool:
    """
    Validate that embeddings configuration is properly set.

    Returns:
        True if configuration is valid, False otherwise
    """
    required_vars = [
        settings.AZURE_OPENAI_ENDPOINT,
        settings.AZURE_OPENAI_API_KEY,
        settings.AZURE_OPENAI_API_VERSION,
        settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    ]

    return all(var is not None for var in required_vars)


def get_embeddings_info() -> dict:
    """
    Get information about available embeddings configurations.

    Returns:
        Dictionary with embeddings configuration information
    """
    return {
        "endpoint": settings.AZURE_OPENAI_ENDPOINT,
        "api_version": settings.AZURE_OPENAI_API_VERSION,
        "deployment_name": settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        "config_valid": validate_embeddings_config(),
    }
