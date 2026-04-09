"""
LLM Factory for the Emergency Medicine Triage RAG System.

This module provides factory functions to initialize different LLM clients
including DeepSeek, Azure OpenAI (GPT-4o), and Anthropic Claude.
"""

from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    ChatDeepSeek = None

from ..config.settings import settings


def initialize_deepseek(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    use_native_client: bool = True,
) -> ChatOpenAI | ChatDeepSeek:
    """
    Initialize DeepSeek client using LangChain.

    Args:
        api_key: DeepSeek API key (defaults to settings)
        base_url: DeepSeek API base URL (defaults to settings)
        model_name: Model name to use (defaults to settings)
        temperature: Temperature for generation (defaults to settings)
        use_native_client: If True and available, use ChatDeepSeek instead of ChatOpenAI

    Returns:
        ChatDeepSeek or ChatOpenAI: Initialized DeepSeek client
    """
    if use_native_client and DEEPSEEK_AVAILABLE:
        # Use native ChatDeepSeek client for better integration
        return ChatDeepSeek(
            model=model_name or settings.DEEPSEEK_MODEL_NAME,
            api_key=api_key or settings.DEEPSEEK_API_KEY,
            temperature=(
                temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
            ),
        )
    else:
        # Fallback to ChatOpenAI with base_url
        return ChatOpenAI(
            model=model_name or settings.DEEPSEEK_MODEL_NAME,
            api_key=api_key or settings.DEEPSEEK_API_KEY,
            base_url=base_url or settings.DEEPSEEK_BASE_URL,
            temperature=(
                temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
            ),
        )


def initialize_azure_openai(
    azure_deployment_name: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    temperature: Optional[float] = None,
) -> AzureChatOpenAI:
    """
    Initialize Azure OpenAI client.

    Args:
        azure_deployment_name: Azure deployment name (defaults to settings)
        azure_endpoint: Azure endpoint URL (defaults to settings)
        azure_api_key: Azure API key (defaults to settings)
        azure_api_version: Azure API version (defaults to settings)
        temperature: Temperature for generation (defaults to settings)

    Returns:
        AzureChatOpenAI: Initialized Azure OpenAI client
    """
    return AzureChatOpenAI(
        azure_deployment=azure_deployment_name
        or settings.AZURE_OPENAI_LLM_DEPLOYMENT_NAME,
        azure_endpoint=azure_endpoint or settings.AZURE_OPENAI_ENDPOINT,
        api_key=azure_api_key or settings.AZURE_OPENAI_API_KEY,
        api_version=azure_api_version or settings.AZURE_OPENAI_API_VERSION,
        temperature=(
            temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        ),
    )


def initialize_claude(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = None,
) -> ChatAnthropic:
    """
    Initialize Claude client using LangChain.

    Args:
        api_key: Anthropic API key (defaults to settings)
        model_name: Model name to use (defaults to settings)
        temperature: Temperature for generation (defaults to settings)
        max_tokens: Maximum tokens to generate (defaults to settings)
        streaming: Whether to use streaming (defaults to settings)

    Returns:
        ChatAnthropic: Initialized Claude client
    """
    return ChatAnthropic(
        model=model_name or settings.ANTHROPIC_MODEL_NAME,
        anthropic_api_key=api_key or settings.ANTHROPIC_API_KEY,
        temperature=(
            temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        ),
        max_tokens=max_tokens or settings.DEFAULT_MAX_TOKENS,
        streaming=streaming if streaming is not None else settings.DEFAULT_STREAMING,
    )


def create_llm(
    model_type: str, **kwargs
) -> ChatOpenAI | AzureChatOpenAI | ChatAnthropic:
    """
    Factory function to create an LLM client based on model type.

    Args:
        model_type: Type of model to create ("deepseek", "gpt4o", "claude")
        **kwargs: Additional arguments to pass to the specific initializer

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()

    if model_type in ["deepseek", "deepseekv3"]:
        return initialize_deepseek(**kwargs)
    elif model_type in ["gpt4o", "gpt-4o", "azure"]:
        return initialize_azure_openai(**kwargs)
    elif model_type in ["claude", "claude-3-7", "anthropic"]:
        return initialize_claude(**kwargs)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: deepseek, gpt4o, claude"
        )


def get_model_name_for_db(model_type: str) -> str:
    """
    Get the model name used for database paths.

    Args:
        model_type: Type of model

    Returns:
        Model name for database paths
    """
    model_type = model_type.lower()

    if model_type in ["deepseek", "deepseekv3"]:
        return "deepseekv3"
    elif model_type in ["gpt4o", "gpt-4o", "azure"]:
        return "gpt-4o"
    elif model_type in ["claude", "claude-3-7", "anthropic"]:
        return "claude-3-7"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_available_models() -> dict:
    """
    Get information about available models.

    Returns:
        Dictionary with model information
    """
    return {
        "deepseek": {
            "aliases": ["deepseek", "deepseekv3"],
            "db_name": "deepseekv3",
            "description": "DeepSeek-v3 model",
        },
        "gpt4o": {
            "aliases": ["gpt4o", "gpt-4o", "azure"],
            "db_name": "gpt-4o",
            "description": "Azure OpenAI GPT-4o model",
        },
        "claude": {
            "aliases": ["claude", "claude-3-7", "anthropic"],
            "db_name": "claude-3-7",
            "description": "Anthropic Claude 3.7 Sonnet model",
        },
    }
