"""
Validators for different question types.

Each validator takes the actual result and expected result,
and returns (is_valid, message).
"""

import math
from typing import Dict, List, Tuple


def validate_exact_match(
    actual: Dict, expected: Dict, fields: List[str] | None = None
) -> Tuple[bool, str]:
    """
    Validate that actual matches expected for specified fields.

    Args:
        actual: Actual result from function
        expected: Expected result
        fields: List of fields to compare. If None, compare all expected fields.

    Returns:
        (is_valid, message)
    """
    if fields is None:
        fields = list(expected.keys())

    differences = []
    for field in fields:
        if field not in actual:
            differences.append(f"Missing field: {field}")
        elif actual[field] != expected[field]:
            differences.append(f"Field '{field}': expected {expected[field]}, got {actual[field]}")

    if differences:
        return False, "; ".join(differences)
    return True, "OK"


def validate_numeric_tolerance(
    actual: Dict, expected: Dict, fields: List[str], tolerance: float = 1e-5
) -> Tuple[bool, str]:
    """
    Validate numeric fields with tolerance.

    Args:
        actual: Actual result
        expected: Expected result
        fields: Fields to compare
        tolerance: Acceptable difference

    Returns:
        (is_valid, message)
    """
    differences = []

    for field in fields:
        if field not in actual:
            differences.append(f"Missing field: {field}")
            continue

        actual_val = actual[field]
        expected_val = expected[field]

        if isinstance(expected_val, list):
            if not isinstance(actual_val, list):
                differences.append(f"Field '{field}': expected list, got {type(actual_val)}")
                continue
            if len(actual_val) != len(expected_val):
                differences.append(
                    f"Field '{field}': length mismatch ({len(actual_val)} vs {len(expected_val)})"
                )
                continue
            for i, (a, e) in enumerate(zip(actual_val, expected_val)):
                if abs(float(a) - float(e)) > tolerance:
                    differences.append(f"Field '{field}[{i}]': {a} != {e} (tolerance: {tolerance})")
        else:
            if abs(float(actual_val) - float(expected_val)) > tolerance:
                differences.append(
                    f"Field '{field}': {actual_val} != {expected_val} (tolerance: {tolerance})"
                )

    if differences:
        return False, "; ".join(differences[:5])  # Limit messages
    return True, "OK"


# ==================== Question-specific validators ====================


def validate_prime_number_generator(
    actual: Dict, expected: Dict, validation_type: str | None = None
) -> Tuple[bool, str]:
    """Validate prime number generator output."""
    if validation_type == "count_only":
        # Only check count for performance tests
        if "count" not in actual:
            return False, "Missing 'count' field"
        if actual["count"] != expected["count"]:
            return False, f"Count mismatch: {actual['count']} != {expected['count']}"
        return True, "OK"
    return validate_exact_match(actual, expected, ["count", "data"])


def validate_perfect_number_generator(
    actual: Dict, expected: Dict, validation_type: str | None = None
) -> Tuple[bool, str]:
    """Validate perfect number generator output."""
    if validation_type == "count_only":
        if "count" not in actual:
            return False, "Missing 'count' field"
        if actual["count"] != expected["count"]:
            return False, f"Count mismatch: {actual['count']} != {expected['count']}"
        return True, "OK"
    return validate_exact_match(actual, expected, ["count", "data"])


def validate_car_position(
    actual: Dict, expected: Dict, tolerance_or_validation=1e-5
) -> Tuple[bool, str]:
    """Validate car position collision times."""
    # Handle validation type or tolerance
    if isinstance(tolerance_or_validation, str):
        validation_type = tolerance_or_validation
        if validation_type == "length_check":
            if "answer" not in actual:
                return False, "Missing 'answer' field"
            # Just check that answer has correct length
            expected_length = expected.get("expected_length", len(expected.get("answer", [])))
            if len(actual["answer"]) != expected_length:
                return (
                    False,
                    f"Length mismatch: {len(actual['answer'])} != {expected_length}",
                )
            return True, "OK"
        return True, "OK"

    tolerance = tolerance_or_validation
    return validate_numeric_tolerance(actual, expected, ["answer"], tolerance)


def validate_median_of_two_array(
    actual: Dict, expected: Dict, tolerance: float = 1e-5
) -> Tuple[bool, str]:
    """Validate median of two arrays."""
    return validate_numeric_tolerance(actual, expected, ["median"], tolerance)


def validate_minimal_cost_split(
    actual: Dict, expected: Dict | None = None, input_data: Dict | None = None
) -> Tuple[bool, str]:
    """
    Validate minimal cost split.

    Checks:
    1. Output has 5 arrays (array_1 to array_5)
    2. Arrays are contiguous and cover the entire input
    3. Cost is sum of first elements
    """
    # Check structure
    required_arrays = ["array_1", "array_2", "array_3", "array_4", "array_5"]
    for arr_name in required_arrays:
        if arr_name not in actual:
            return False, f"Missing {arr_name}"

    if "cost" not in actual:
        return False, "Missing cost field"

    # Check arrays cover input
    if input_data:
        input_array = input_data.get("array", [])
        combined = []
        for arr_name in required_arrays:
            combined.extend(actual[arr_name])

        if combined != input_array:
            return False, "Arrays don't form the original input array"

    # Verify cost calculation
    cost = 0
    for arr_name in required_arrays:
        arr = actual[arr_name]
        if arr:  # Non-empty array
            cost += arr[0]

    if actual["cost"] != cost:
        return False, f"Cost mismatch: reported {actual['cost']}, calculated {cost}"

    # Check expected cost if provided
    if expected and "expected_cost" in expected:
        if actual["cost"] != expected["expected_cost"]:
            return (
                False,
                f"Cost {actual['cost']} != expected {expected['expected_cost']}",
            )

    return True, "OK"


def validate_pagerank(actual: Dict, expected: Dict | None = None) -> Tuple[bool, str]:
    """
    Validate PageRank output.

    Checks structure and reasonable values.
    """
    # Check for pagerank value (could be various key names)
    pagerank_keys = ["pagerank", "pagerank_value", "first_node_pagerank"]
    pagerank_found = False
    pagerank_value = None

    for key in pagerank_keys:
        if key in actual:
            pagerank_found = True
            pagerank_value = actual[key]
            break

    # Also check nested structure
    if not pagerank_found:
        for key, value in actual.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                pagerank_found = True
                pagerank_value = value
                break

    if not pagerank_found:
        return False, "No PageRank value found"

    # Validate PageRank is in [0, 1]
    if pagerank_value is not None:
        if not (0 <= pagerank_value <= 1):
            return False, f"PageRank value {pagerank_value} not in [0, 1]"

    # Check timing fields (optional but good to have)
    timing_keys = ["graph_generating_time", "compute_time", "time", "timing"]
    timing_found = any(k in actual or k in str(actual) for k in timing_keys)

    return True, "OK"


# Validator registry
VALIDATORS = {
    "prime_number_generator": validate_prime_number_generator,
    "perfect_number_generator": validate_perfect_number_generator,
    "car_position": validate_car_position,
    "car_positon": validate_car_position,  # Handle typo in folder names
    "median_of_two_array": validate_median_of_two_array,
    "minimal_cost_split": validate_minimal_cost_split,
    "pagerank": validate_pagerank,
}


def get_validator(question_name: str):
    """Get validator function for a question type."""
    # Normalize name
    name = question_name.lower().replace("-", "_").replace(" ", "_")
    return VALIDATORS.get(name)
