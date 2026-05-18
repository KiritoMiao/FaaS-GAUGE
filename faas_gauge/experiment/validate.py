"""Batch validation orchestration."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from faas_gauge.experiment.classify import classify_function_status, is_performance_test
from faas_gauge.store import ExperimentStore
from faas_gauge.validator import FunctionValidator
from faas_gauge.validator.utils import sanitize_large_output
from faas_gauge.validator.validators import get_validator


def _load_test_data(question_name: str, data_dir: Path) -> Optional[dict]:
    """Load test data for a question type from data/test_data/."""
    test_data_dir = data_dir / "test_data"

    names_to_try = [
        question_name,
        question_name.replace("-", "_"),
        question_name.replace("_", "-"),
    ]

    for name in names_to_try:
        test_file = test_data_dir / f"{name}.json"
        if test_file.exists():
            with open(test_file, "r", encoding="utf-8") as handle:
                loaded: dict[Any, Any] = json.load(handle)
                return loaded
    return None


def _expand_tests_for_performance_sizes(
    tests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand performance tests into multiple size variants."""
    expanded: list[dict[str, Any]] = []

    for original in tests:
        test = dict(original or {})
        test_name = test.get("name", "unnamed")

        if not is_performance_test(test_name):
            expanded.append(test)
            continue

        size_tests = test.get("size_tests")
        if isinstance(size_tests, list) and size_tests:
            for idx, case in enumerate(size_tests, start=1):
                case = dict(case or {})
                merged = dict(test)
                merged.pop("size_tests", None)
                case_input = case.pop("input", None)
                merged.update(case)
                if case_input is not None:
                    merged["input"] = case_input
                label = merged.get("size_label") or merged.get("name") or f"size_{idx}"
                merged["name"] = f"{test_name}[{label}]"
                expanded.append(merged)
            continue

        sizes = test.get("sizes")
        size_key = test.get("size_key")
        if isinstance(sizes, list) and sizes and isinstance(size_key, str) and size_key:
            expected_by_size = test.get("expected_by_size") or {}
            for size in sizes:
                merged = dict(test)
                merged.pop("sizes", None)
                merged.pop("size_key", None)
                merged.pop("expected_by_size", None)
                base_input = dict(test.get("input") or {})
                base_input[size_key] = size
                merged["input"] = base_input
                expected = expected_by_size.get(str(size), expected_by_size.get(size))
                if expected is not None:
                    merged["expected"] = expected
                merged["name"] = f"{test_name}[{size_key}={size}]"
                expanded.append(merged)
            continue

        expanded.append(test)

    return expanded


def validate_code(
    code_path: str,
    runtime: str,
    iterations: int,
    max_runtime: int,
    test_data: Optional[dict] = None,
    performance_mode: str = "loose",
    aws_memory_size: int = 1769,
    reuse_deployed_function: bool = True,
    config_path: str | None = None,
    max_total_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate a single code file."""
    question = None
    parts = code_path.split(os.sep)
    for part in parts:
        if "+" in part:
            segments = part.split("+")
            if len(segments) >= 2:
                question = segments[1]
                break

    result: dict[str, Any] = {
        "code_path": code_path,
        "question": question,
        "runtime": runtime,
        "iterations_per_test": iterations,
        "max_runtime": max_runtime,
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total_executions": 0,
        },
    }

    validator_func = get_validator(question) if question else None

    tests: list[dict[str, Any]]
    if test_data and "tests" in test_data:
        tests = list(test_data["tests"])
    else:
        tests = [{"name": "default", "input": {}, "expected": None}]
    tests = _expand_tests_for_performance_sizes(tests)

    try:
        with open(code_path, "r", encoding="utf-8", errors="replace") as handle:
            code_content = handle.read().strip()
    except Exception as exc:
        code_content = ""
        read_error = str(exc)
    else:
        read_error = ""

    if not code_content:
        for test in tests:
            result["tests"].append(
                {
                    "name": test.get("name", "unnamed"),
                    "input": test.get("input", {}),
                    "status": "error",
                    "message": read_error or "Empty code file",
                    "execution_results": [],
                    "iterations": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                }
            )
            result["summary"]["errors"] += 1
            result["summary"]["total_tests"] += 1
        status, reason = classify_function_status(result)
        result["function_status"] = status
        result["function_status_reason"] = reason
        return result

    fv = None
    try:
        fv = FunctionValidator(
            code_path=code_path,
            runtime=runtime,
            test_input={},
            timeout=max_runtime,
            aws_memory_size=aws_memory_size,
            reuse_deployed_function=reuse_deployed_function,
            config_path=config_path,
        )
        if not fv.build():
            build_error = getattr(
                getattr(fv, "_runner", None), "_build_error", "Build failed"
            )
            for test in tests:
                result["tests"].append(
                    {
                        "name": test.get("name", "unnamed"),
                        "input": test.get("input", {}),
                        "status": "error",
                        "message": f"Failed to build validator: {build_error}",
                        "execution_results": [],
                        "iterations": 0,
                        "successful_runs": 0,
                        "failed_runs": 0,
                    }
                )
                result["summary"]["errors"] += 1
                result["summary"]["total_tests"] += 1
            status, reason = classify_function_status(result)
            result["function_status"] = status
            result["function_status_reason"] = reason
            return result
    except Exception as exc:
        for test in tests:
            result["tests"].append(
                {
                    "name": test.get("name", "unnamed"),
                    "input": test.get("input", {}),
                    "status": "error",
                    "message": f"Failed to create validator: {exc}",
                    "execution_results": [],
                    "iterations": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                }
            )
            result["summary"]["errors"] += 1
            result["summary"]["total_tests"] += 1
        status, reason = classify_function_status(result)
        result["function_status"] = status
        result["function_status_reason"] = reason
        return result

    for test in tests:
        test_name = test.get("name", "unnamed")
        test_input = test.get("input", {}) or {}
        test_result: dict[str, Any] = {
            "name": test_name,
            "input": test_input,
            "status": "unknown",
            "message": "",
            "execution_results": [],
        }

        performance_test = is_performance_test(test_name)
        default_iters = iterations if performance_test else 1
        test_iterations = int(test.get("iterations", default_iters) or default_iters)
        exclude_result = bool(test.get("exclude_result", performance_test))

        perf_budget = max_total_seconds if performance_test else None
        try:
            run_result = fv.run(
                iterations=test_iterations,
                input_data=test_input,
                max_total_seconds=perf_budget,
            )
        except Exception as exc:
            test_result["status"] = "error"
            test_result["message"] = f"Execution failed: {exc}"
            result["summary"]["errors"] += 1
            result["tests"].append(test_result)
            result["summary"]["total_tests"] += 1
            continue

        for exec_result in run_result.executions:
            exec_data: dict[str, Any] = {
                "iteration": exec_result.iteration,
                "success": exec_result.success,
                "execution_time_ms": exec_result.execution_time_ms,
                "timestamp": exec_result.timestamp,
                "error": exec_result.error,
                "max_memory_used_mb": exec_result.max_memory_used_mb,
                "cloudwatch_request_id": exec_result.cloudwatch_request_id,
            }
            if exec_result.output_too_large:
                exec_data["output_too_large"] = True
            if not exclude_result:
                safe_result, _ = sanitize_large_output(exec_result.result)
                safe_output, _ = sanitize_large_output(exec_result.output)
                exec_data["result"] = safe_result
                exec_data["output"] = safe_output
            else:
                safe_output, _ = sanitize_large_output(exec_result.output)
                exec_data["output"] = safe_output
            test_result["execution_results"].append(exec_data)

        if run_result.failed_runs > 0:
            test_result["status"] = "error"
            test_result["message"] = (
                f"{run_result.failed_runs}/{run_result.iterations} executions failed"
            )
            result["summary"]["errors"] += 1
        else:
            first_success = next((e for e in run_result.executions if e.success), None)
            if first_success is None:
                test_result["status"] = "error"
                test_result["message"] = "No successful execution"
                result["summary"]["errors"] += 1
            else:
                actual = first_success.result
                if performance_test and performance_mode == "loose":
                    if actual is None and first_success.output is None:
                        test_result["status"] = "failed"
                        test_result["message"] = (
                            "Performance execution produced no output"
                        )
                        result["summary"]["failed"] += 1
                    else:
                        test_result["status"] = "passed"
                        test_result["message"] = (
                            "Performance execution successful (loose validation)"
                        )
                        result["summary"]["passed"] += 1
                elif validator_func and test.get("expected") is not None:
                    expected = test["expected"]
                    tolerance = test.get("tolerance", 1e-5)
                    validation_type = test.get("validation")
                    try:
                        if validation_type:
                            is_valid, msg = validator_func(
                                actual, expected, validation_type
                            )
                        elif "tolerance" in test:
                            is_valid, msg = validator_func(actual, expected, tolerance)
                        else:
                            is_valid, msg = validator_func(actual, expected)
                    except TypeError:
                        try:
                            is_valid, msg = validator_func(actual, expected)
                        except TypeError:
                            is_valid, msg = validator_func(actual)
                    if is_valid:
                        test_result["status"] = "passed"
                        test_result["message"] = msg
                        result["summary"]["passed"] += 1
                    else:
                        test_result["status"] = "failed"
                        test_result["message"] = msg
                        result["summary"]["failed"] += 1
                else:
                    test_result["status"] = "passed"
                    test_result["message"] = "Execution successful"
                    result["summary"]["passed"] += 1

        test_result["iterations"] = run_result.iterations
        test_result["successful_runs"] = run_result.successful_runs
        test_result["failed_runs"] = run_result.failed_runs
        result["tests"].append(test_result)
        result["summary"]["total_tests"] += 1
        result["summary"]["total_executions"] += len(
            test_result.get("execution_results", [])
        )

    if fv is not None:
        try:
            fv.cleanup()
        except Exception:
            pass

    status, reason = classify_function_status(result)
    result["function_status"] = status
    result["function_status_reason"] = reason
    return result


def validate_batch(
    test_group: str,
    iterations: int = 1,
    runtime: str = "local",
    max_runtime: int = 30,
    question: str | None = None,
    performance_mode: str = "loose",
    aws_memory_size: int = 1769,
    reuse_deployed_function: bool = True,
    data_dir: str | Path = "data",
    config_path: str | None = None,
    max_total_seconds: float | None = None,
    skip_resume: bool = False,
) -> dict[str, Any]:
    """Run batch validation on experiments in a test group."""
    data_dir = Path(data_dir)
    store = ExperimentStore(data_dir=data_dir)

    experiments = store.list_experiments(test_group=test_group)
    if question:
        experiments = [
            e
            for e in experiments
            if question.lower() in e.get("question_name", "").lower()
        ]

    # ── Resume logic: skip iterations already validated in prior batches ──
    already_validated: set[tuple[str, int]] = set()
    if not skip_resume:
        try:
            all_batches = store.list_validations()
            for prior in all_batches:
                pb = prior if isinstance(prior, dict) else {}
                if (
                    pb.get("test_group") == test_group
                    and pb.get("runtime") == runtime
                    and pb.get("question_filter") == question
                    and pb.get("iterations_per_test") == iterations
                ):
                    prior_id = pb.get("id", "")
                    for run in store.read_validation_runs(prior_id):
                        eid = run.get("experiment_id", "")
                        it = run.get("experiment_iteration") or run.get("iteration", 0)
                        if eid and it:
                            already_validated.add((eid, int(it)))
        except Exception:
            pass  # If listing fails, just re-validate everything

    batch_id = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    store.create_validation_batch(
        {
            "id": batch_id,
            "test_group": test_group,
            "runtime": runtime,
            "iterations_per_test": iterations,
            "max_runtime": max_runtime,
            "question_filter": question,
        }
    )

    summary: dict[str, Any] = {
        "total_experiments": len(experiments),
        "completed": 0,
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
    }

    for exp_meta in experiments:
        exp_id = exp_meta["id"]
        question_name = exp_meta.get("question_name", "")
        test_data = _load_test_data(question_name, data_dir)

        for iteration_record in store.read_iterations(exp_id):
            code = iteration_record.get("python_code", "")
            if not code:
                continue

            iteration_num = iteration_record.get("iteration", 0)

            # Skip if already validated in a prior batch
            if (exp_id, int(iteration_num)) in already_validated:
                continue

            tmp_dir = tempfile.mkdtemp()
            code_path = os.path.join(tmp_dir, "code.py")
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(code)

            try:
                result = validate_code(
                    code_path=code_path,
                    runtime=runtime,
                    iterations=iterations,
                    max_runtime=max_runtime,
                    test_data=test_data,
                    performance_mode=performance_mode,
                    aws_memory_size=aws_memory_size,
                    reuse_deployed_function=reuse_deployed_function,
                    config_path=config_path,
                    max_total_seconds=max_total_seconds,
                )

                store.append_validation_run(
                    batch_id,
                    {
                        "experiment_id": exp_id,
                        "experiment_iteration": iteration_num,
                        "iteration": iteration_num,
                        "question": question_name,
                        "function_status": result.get("function_status", "unknown"),
                        "function_status_reason": result.get(
                            "function_status_reason", ""
                        ),
                        "summary": result.get("summary", {}),
                        "tests": result.get("tests", []),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

                summary["completed"] += 1
                summary["total_tests"] += result["summary"]["total_tests"]
                summary["passed"] += result["summary"]["passed"]
                summary["failed"] += result["summary"]["failed"]
                summary["errors"] += result["summary"]["errors"]

            except Exception as exc:
                store.append_validation_run(
                    batch_id,
                    {
                        "experiment_id": exp_id,
                        "iteration": iteration_num,
                        "function_status": "error",
                        "function_status_reason": str(exc),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                summary["errors"] += 1

            finally:
                try:
                    os.remove(code_path)
                    os.rmdir(tmp_dir)
                except Exception:
                    pass

    store.update_validation_batch(
        batch_id,
        {"summary": summary, "completed_at": datetime.now().isoformat()},
    )

    return {"batch_id": batch_id, "summary": summary}
