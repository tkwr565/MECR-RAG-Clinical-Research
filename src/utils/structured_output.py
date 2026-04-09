"""
Utility functions for handling structured outputs from LLMs.

This module provides helper functions for token tracking and retry logic
when using structured outputs with Pydantic schemas.
"""

import json
from typing import Callable, Any
from pydantic import ValidationError

from ..models.schemas import TokenUsageMetrics
from ..data.state import GraphState


def get_token_usage(response) -> TokenUsageMetrics:
    """
    Extract token usage from LLM response if available.

    Args:
        response: LLM response object

    Returns:
        TokenUsageMetrics object with token counts
    """
    try:
        # For OpenAI-compatible APIs, token usage is in response_metadata
        if (
            hasattr(response, "response_metadata")
            and "token_usage" in response.response_metadata
        ):
            usage = response.response_metadata["token_usage"]
            return TokenUsageMetrics(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
    except Exception as e:
        print(f"Could not extract token usage: {e}")

    return TokenUsageMetrics()


def execute_structured_node_with_retry(
    node_func: Callable[[GraphState], GraphState],
    state: GraphState,
    node_name: str,
    max_retries: int = 3,
) -> GraphState:
    """
    Execute a node with retry logic for malformed structured responses.

    This function wraps node execution with automatic retry on validation errors,
    which can occur when LLMs return malformed JSON or invalid structured outputs.

    Args:
        node_func: The node function to execute
        state: Current graph state
        node_name: Name of the node (for logging)
        max_retries: Maximum number of retry attempts

    Returns:
        Updated state or state with failure flag
    """
    for attempt in range(max_retries):
        try:
            print(f"\n=== {node_name}: Attempt {attempt + 1}/{max_retries} ===")
            result = node_func(state)
            print(f"{node_name} succeeded on attempt {attempt + 1}")
            return result
        except (ValidationError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"{node_name} attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Mark case as failed after all retries exhausted
                print(f"{node_name} FAILED after {max_retries} retries")
                failed_state = state.copy()
                failed_state["processing_failed"] = True
                failed_state["failure_reason"] = (
                    f"{node_name} failed after {max_retries} retries: {str(e)}"
                )
                failed_state["failed_node"] = node_name
                return failed_state
        except Exception as e:
            print(f"{node_name} attempt {attempt + 1} failed with unexpected error: {e}")
            if attempt == max_retries - 1:
                # Propagate unexpected errors after final retry
                print(f"{node_name} FAILED with unexpected error after {max_retries} retries")
                failed_state = state.copy()
                failed_state["processing_failed"] = True
                failed_state["failure_reason"] = (
                    f"{node_name} failed with unexpected error: {str(e)}"
                )
                failed_state["failed_node"] = node_name
                return failed_state

    # This should not be reached, but return failed state as fallback
    failed_state = state.copy()
    failed_state["processing_failed"] = True
    failed_state["failure_reason"] = f"{node_name} failed unexpectedly"
    failed_state["failed_node"] = node_name
    return failed_state


def check_processing_failed(state: GraphState) -> bool:
    """
    Check if processing has failed in the current state.

    Args:
        state: Current graph state

    Returns:
        True if processing failed, False otherwise
    """
    if isinstance(state, dict):
        return state.get("processing_failed", False)
    return getattr(state, "processing_failed", False)


def get_failure_info(state: GraphState) -> dict:
    """
    Extract failure information from state.

    Args:
        state: Current graph state

    Returns:
        Dictionary with failure information
    """
    if isinstance(state, dict):
        return {
            "failed": state.get("processing_failed", False),
            "reason": state.get("failure_reason", ""),
            "failed_node": state.get("failed_node", ""),
        }

    return {
        "failed": getattr(state, "processing_failed", False),
        "reason": getattr(state, "failure_reason", ""),
        "failed_node": getattr(state, "failed_node", ""),
    }
