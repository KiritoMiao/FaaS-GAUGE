"""Function status classification."""

from typing import Any


def is_performance_test(test_name: str) -> bool:
    """Return True when a test name indicates performance validation."""
    return str(test_name or "").lower().startswith("performance")


def _contains_any(text: str, needles: list[str]) -> bool:
    """Case-insensitive substring match against any needle."""
    hay = str(text or "").lower()
    return any(needle in hay for needle in needles)


def classify_function_status(validation_result: dict[str, Any]) -> tuple[str, str]:
    """
    Classify one generated code result into a single status label.

    Args:
        validation_result: Dict with "tests" key containing test results.

    Returns:
        Tuple of (status_label, reason_message).

    Supported labels:
      success, syntax-error, functional-error, timeout, out-of-memory, scaling-bug
    """
    tests = validation_result.get("tests") or []
    if not tests:
        return "functional-error", "No tests were executed"

    all_messages: list[str] = []
    for test in tests:
        msg = test.get("message")
        if msg:
            all_messages.append(str(msg))
        for exec_row in test.get("execution_results", []) or []:
            err = exec_row.get("error")
            if err:
                all_messages.append(str(err))

    combined = " | ".join(all_messages).lower()

    syntax_markers = [
        "syntaxerror",
        "indentationerror",
        "taberror",
        "invalid syntax",
        "failed to parse",
    ]
    oom_markers = [
        "memoryerror",
        "out of memory",
        "outofmemory",
        "oom",
        "runtime exited with error: signal: killed",
        "process exited before completing request",
    ]
    timeout_markers = [
        "timed out",
        "time out",
        "timeout",
        "task timed out",
        "function execution timed out",
    ]

    if _contains_any(combined, syntax_markers):
        return "syntax-error", "Build or execution failed due to syntax/parsing issues"

    if _contains_any(combined, oom_markers):
        return "out-of-memory", "Execution failed due to memory exhaustion"

    perf_tests = [t for t in tests if is_performance_test(t.get("name", ""))]
    non_perf_tests = [t for t in tests if not is_performance_test(t.get("name", ""))]

    perf_failures = [t for t in perf_tests if t.get("status") in {"failed", "error"}]
    non_perf_failures = [t for t in non_perf_tests if t.get("status") in {"failed", "error"}]
    non_perf_all_passed = bool(non_perf_tests) and all(
        t.get("status") == "passed" for t in non_perf_tests
    )

    if perf_failures and non_perf_all_passed:
        return (
            "scaling-bug",
            "Functional tests passed but one or more performance steps failed",
        )

    if _contains_any(combined, timeout_markers):
        return "timeout", "Execution exceeded runtime limits"

    if non_perf_failures or perf_failures:
        return (
            "functional-error",
            "Output did not satisfy one or more validation checks",
        )

    if all(t.get("status") == "passed" for t in tests):
        return "success", "All validation tests passed"

    return "functional-error", "Validation completed with non-passing status"
