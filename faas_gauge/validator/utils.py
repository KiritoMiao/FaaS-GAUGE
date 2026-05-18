"""
Utility functions for function_validator package.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import cast
from typing import Any, Dict, Optional, Union

TOO_LARGE_OUTPUT_FLAG = "____TOO_LARGE_OUTPUT____"
MAX_OUTPUT_BYTES = 5 * 1024 * 1024


def _json_size_bytes(value: Any) -> int:
    """Estimate serialized JSON byte size for a value."""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="replace"))


def sanitize_large_output(
    value: Any,
    max_bytes: int = MAX_OUTPUT_BYTES,
    flag: str = TOO_LARGE_OUTPUT_FLAG,
) -> tuple[Any, bool]:
    """
    Replace oversized output blocks with a sentinel flag.

    Returns:
        (sanitized_value, was_truncated)
    """
    if value is None:
        return value, False

    if _json_size_bytes(value) <= max_bytes:
        return value, False

    if isinstance(value, dict):
        sanitized_dict: dict[Any, Any] = {}
        replaced = False
        for key, item in value.items():
            if _json_size_bytes(item) > max_bytes:
                sanitized_dict[key] = flag
                replaced = True
            else:
                sanitized_dict[key] = item

        if not replaced or _json_size_bytes(sanitized_dict) > max_bytes:
            return {"__validator_output__": flag}, True
        return sanitized_dict, True

    if isinstance(value, list):
        sanitized_list: list[Any] = []
        replaced = False
        for item in value:
            if _json_size_bytes(item) > max_bytes:
                sanitized_list.append(flag)
                replaced = True
            else:
                sanitized_list.append(item)

        if not replaced or _json_size_bytes(sanitized_list) > max_bytes:
            return flag, True
        return sanitized_list, True

    return flag, True


def load_json_input(input_data: Union[str, Dict, Path]) -> Dict[str, Any]:
    """
    Load JSON input from various sources.

    Args:
        input_data: Can be:
            - A dictionary (returned as-is)
            - A JSON string
            - A path to a JSON file

    Returns:
        Parsed dictionary.
    """
    if isinstance(input_data, dict):
        return input_data

    if isinstance(input_data, Path):
        input_data = str(input_data)

    if isinstance(input_data, str):
        # Check if it's a file path
        if os.path.isfile(input_data):
            with open(input_data, "r") as f:
                loaded: Dict[str, Any] = json.load(f)
                return loaded
        else:
            # Try to parse as JSON string
            try:
                loaded = json.loads(input_data)
                if not isinstance(loaded, dict):
                    raise ValueError(f"JSON input must decode to an object: {input_data}")
                return cast(Dict[str, Any], loaded)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string or file not found: {input_data}")

    raise TypeError(f"Expected dict, str, or Path, got {type(input_data)}")


def generate_result_filename(prefix: str = "runner") -> str:
    """
    Generate a result filename with timestamp.

    Args:
        prefix: Prefix for the filename.

    Returns:
        Filename in format: {prefix}+YYYYMMDD_HHMM.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{prefix}+{timestamp}.json"


def save_results(results: Dict[str, Any], code_path: str, filename: Optional[str] = None) -> str:
    """
    Save results to a JSON file in the code's directory.

    Args:
        results: Results dictionary to save.
        code_path: Path to the code file (results saved in same directory).
        filename: Optional custom filename. If None, generates timestamped name.

    Returns:
        Path to the saved file.
    """
    if filename is None:
        filename = generate_result_filename()

    # Get the directory containing the code
    code_dir = os.path.dirname(os.path.abspath(code_path))
    output_path = os.path.join(code_dir, filename)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return output_path


def extract_function_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the actual function result from SAAF-wrapped output.

    Handles multiple response formats:
    1. Direct result: {"key": "value", ...} + SAAF attributes
    2. API Gateway style: {"statusCode": 200, "body": "{...}"} + SAAF attributes
    3. Body as dict: {"statusCode": 200, "body": {...}} + SAAF attributes

    Args:
        result: Full result including SAAF attributes.

    Returns:
        Extracted function result (without SAAF attributes).
    """
    # First filter out SAAF attributes
    filtered = filter_saaf_attributes(result)

    # Check for API Gateway-style response with body
    if "body" in filtered and "statusCode" in filtered:
        body = filtered.get("body")

        # If body is a JSON string, parse it
        if isinstance(body, str):
            try:
                parsed_body = json.loads(body)
                if isinstance(parsed_body, dict):
                    return cast(Dict[str, Any], parsed_body)
            except (json.JSONDecodeError, TypeError):
                pass

        # If body is already a dict, return it
        if isinstance(body, dict):
            return body

    # Return filtered result as-is
    return filtered


# SAAF attribute keys to filter out
SAAF_KEYS = {
    # Core SAAF attributes
    "version",
    "lang",
    "startTime",
    "endTime",
    "runtime",
    "invocations",
    "initializationTime",
    # Container attributes
    "uuid",
    "newcontainer",
    # CPU attributes
    "cpuType",
    "cpuModel",
    "cpuCores",
    "cpuInfo",
    "architecture",
    "cpuUser",
    "cpuNice",
    "cpuKernel",
    "cpuIdle",
    "cpuIOWait",
    "cpuIrq",
    "cpuSoftIrq",
    "cpuSteal",
    "cpuGuest",
    "cpuGuestNice",
    "cpuUserDelta",
    "cpuNiceDelta",
    "cpuKernelDelta",
    "cpuIdleDelta",
    "cpuIOWaitDelta",
    "cpuIrqDelta",
    "cpuSoftIrqDelta",
    "cpuStealDelta",
    "cpuGuestDelta",
    "cpuGuestNiceDelta",
    "bootTime",
    "cpuPolls",
    # Memory attributes
    "totalMemory",
    "freeMemory",
    "pageFaults",
    "majorPageFaults",
    "pageFaultsDelta",
    "majorPageFaultsDelta",
    # Platform attributes
    "platform",
    "containerID",
    "vmID",
    "functionName",
    "functionMemory",
    "functionRegion",
    # Linux attributes
    "linuxVersion",
    # Timing attributes
    "frameworkRuntime",
    "userRuntime",
    "frameworkRuntimeDeltas",
    # Recommendation attributes
    "availableCPUs",
    "utilizedCPUs",
    "recommendedMemory",
    # Error attributes
    "SAAFCPUDeltaError",
    "SAAFMemoryError",
    "SAAFMemoryDeltaError",
    "SAAFRecommendConfigurationError",
    # Internal success flag (added by SAAF wrapper)
    "success",
}


def filter_saaf_attributes(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out SAAF-specific attributes from a result for comparison.

    SAAF attributes are those added by the Inspector class and include
    timing, platform, CPU, memory metrics, etc.

    Args:
        result: Result dictionary potentially containing SAAF attributes.

    Returns:
        Dictionary with SAAF attributes removed.
    """
    return {k: v for k, v in result.items() if k not in SAAF_KEYS}


def compare_results(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    ignore_saaf: bool = True,
    extract_body: bool = True,
) -> Dict[str, Any]:
    """
    Compare actual results with expected results.

    Args:
        actual: Actual result from function execution.
        expected: Expected result.
        ignore_saaf: If True, ignore SAAF-specific attributes in comparison.
        extract_body: If True, extract result from API Gateway-style body.

    Returns:
        Dictionary with comparison results:
            - match: Boolean indicating if results match
            - differences: List of differences if any
            - actual_extracted: The extracted actual result
            - expected_extracted: The extracted expected result
    """
    if extract_body:
        actual = extract_function_result(actual)
        expected = extract_function_result(expected) if isinstance(expected, dict) else expected
    elif ignore_saaf:
        actual = filter_saaf_attributes(actual)
        expected = filter_saaf_attributes(expected) if isinstance(expected, dict) else expected

    differences = []

    # Check for missing keys
    for key in expected:
        if key not in actual:
            differences.append({"type": "missing_key", "key": key, "expected": expected[key]})

    # Check for extra keys
    for key in actual:
        if key not in expected:
            differences.append({"type": "extra_key", "key": key, "actual": actual[key]})

    # Check for value differences
    for key in expected:
        if key in actual and actual[key] != expected[key]:
            differences.append(
                {
                    "type": "value_mismatch",
                    "key": key,
                    "expected": expected[key],
                    "actual": actual[key],
                }
            )

    return {"match": len(differences) == 0, "differences": differences}


def extract_function_from_code(code: str) -> Optional[str]:
    """
    Extract the main handler function name from code.

    Looks for common patterns like:
    - def handler(...)
    - def yourFunction(...)
    - def lambda_handler(...)
    - def main(...)

    Args:
        code: Python source code.

    Returns:
        Function name if found, None otherwise.
    """
    common_handlers = [
        "handler",
        "yourFunction",
        "lambda_handler",
        "main",
        "function_handler",
        "my_handler",
    ]

    for handler_name in common_handlers:
        pattern = rf"def\s+{handler_name}\s*\("
        if re.search(pattern, code):
            return handler_name

    # Fallback: find first function that takes (request, context) or similar
    pattern = r"def\s+(\w+)\s*\(\s*\w+\s*,\s*\w*\s*\)"
    match = re.search(pattern, code)
    if match:
        return match.group(1)

    return None
