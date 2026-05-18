"""
Utility functions for the faas_gauge AI client
"""

import re
from typing import Any, Dict, Optional


def extract_python_code(text: str) -> str:
    """
    Extract Python code from markdown code blocks

    Args:
        text: Text containing potential code blocks

    Returns:
        str: Extracted Python code
    """
    # Look for ```python code blocks first
    python_pattern = r"```python\s*\n(.*?)\n```"
    matches = re.findall(python_pattern, text, re.DOTALL)

    if matches:
        return "\n".join(matches)

    # Fallback: look for any code blocks
    code_pattern = r"```.*?\n(.*?)\n```"
    matches = re.findall(code_pattern, text, re.DOTALL)

    if matches:
        return "\n".join(matches)

    return ""


def normalize_model_name(model: str, mappings: Optional[Dict[str, str]] = None) -> str:
    """
    Normalize model name using provided mappings

    Args:
        model: Original model name
        mappings: Optional custom mappings

    Returns:
        str: Normalized model name (returns original if no mappings provided)
    """
    if mappings is None:
        # No default mappings to avoid hardcoded values
        return model

    return mappings.get(model, model)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Optional[Dict[str, tuple[float, float]]] = None,
    model_mappings: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """
    Calculate cost based on token usage

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        pricing: Optional custom pricing dictionary
        model_mappings: Optional model name mappings

    Returns:
        Optional[float]: Cost in USD, or None if pricing not available

    Note:
        Cost calculation is disabled by default to avoid hardcoded pricing.
        Provide custom pricing dictionary to enable cost calculation.
    """
    # No default pricing to avoid hardcoded values
    if pricing is None or not pricing:
        return None

    normalized_model = normalize_model_name(model, model_mappings)

    if normalized_model in pricing:
        input_price, output_price = pricing[normalized_model]
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        return float(round(cost, 6))

    return None


def model_fix(model: str) -> str:
    """
    Fix model name for file naming (backward compatibility)

    Args:
        model: Original model name

    Returns:
        str: Fixed model name suitable for filenames
    """
    model_fixed = model.replace("/", "-")
    model_fixed = model_fixed.replace(" ", "-")
    model_fixed = model_fixed.replace(":", "-")
    return model_fixed


def validate_provider_config(config: Dict[str, Any]) -> bool:
    """
    Validate provider configuration

    Args:
        config: Provider configuration dictionary

    Returns:
        bool: True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["api_base", "api_key"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
        if not config[field]:
            raise ValueError(f"Empty value for required field: {field}")

    return True


def format_response_summary(response) -> str:
    """
    Format a response summary for logging/display in the faas_gauge package.

    Args:
        response: AIResponse object

    Returns:
        str: Formatted summary
    """
    summary_parts = [
        f"Provider: {response.provider}",
        f"Model: {response.model}",
        f"Tokens: {response.input_tokens} → {response.output_tokens}",
        f"Time: {response.rtt_sec:.2f}s",
    ]

    if response.cost is not None:
        summary_parts.append(f"Cost: ${response.cost:.6f}")

    if response.tokens_per_second is not None:
        summary_parts.append(f"Speed: {response.tokens_per_second:.1f} tok/s")

    return " | ".join(summary_parts)


def sanitize_api_key(api_key: str, show_chars: int = 4) -> str:
    """
    Sanitize API key for logging (show only first/last few characters)

    Args:
        api_key: Original API key
        show_chars: Number of characters to show at start/end

    Returns:
        str: Sanitized API key
    """
    if len(api_key) <= show_chars * 2:
        return "*" * len(api_key)

    return f"{api_key[:show_chars]}...{api_key[-show_chars:]}"


def parse_error_response(error: Exception) -> Dict[str, Any]:
    """
    Parse error response to extract useful information

    Args:
        error: Exception object

    Returns:
        Dict[str, Any]: Parsed error information
    """
    error_info = {
        "type": type(error).__name__,
        "message": str(error),
        "is_rate_limit": False,
        "is_auth_error": False,
        "is_quota_error": False,
        "retry_after": None,
    }

    error_str = str(error).lower()

    # Check for common error types
    if "rate limit" in error_str or "too many requests" in error_str:
        error_info["is_rate_limit"] = True

    if (
        "unauthorized" in error_str
        or "invalid api key" in error_str
        or "authentication" in error_str
    ):
        error_info["is_auth_error"] = True

    if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
        error_info["is_quota_error"] = True

    # Try to extract retry-after header if present
    if hasattr(error, "response") and error.response:
        headers = getattr(error.response, "headers", {})
        if "retry-after" in headers:
            try:
                error_info["retry_after"] = int(headers["retry-after"])
            except (ValueError, TypeError):
                pass

    return error_info
