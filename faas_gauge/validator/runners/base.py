"""
Base runner class for function execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExecutionResult:
    """Result from a single function execution."""

    success: bool
    """Whether the execution completed without errors."""

    output: Optional[Dict[str, Any]] = None
    """Output from the function (includes SAAF data if injected)."""

    error: Optional[str] = None
    """Error message if execution failed."""

    execution_time_ms: float = 0.0
    """Total execution time in milliseconds."""

    iteration: int = 0
    """Iteration number for this execution."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """Timestamp when execution completed."""

    output_too_large: bool = False
    """Whether output payload was truncated due to size limits."""

    max_memory_used_mb: Optional[float] = None
    """Max memory used for this run (AWS CloudWatch report), if available."""

    cloudwatch_request_id: Optional[str] = None
    """AWS request id associated with the invocation, if available."""

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """
        Extract the actual function result from the output.

        Handles API Gateway-style responses where result is in 'body'.
        Strips SAAF attributes.

        Returns:
            Extracted function result.
        """
        if self.output is None:
            return None

        from ..utils import extract_function_result

        return extract_function_result(self.output)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "result": self.result,  # Include extracted result
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "output_too_large": self.output_too_large,
            "max_memory_used_mb": self.max_memory_used_mb,
            "cloudwatch_request_id": self.cloudwatch_request_id,
        }


@dataclass
class ValidationResult:
    """Result from validation run with multiple iterations."""

    code_path: str
    """Path to the code that was executed."""

    runtime: str
    """Runtime environment used (local/aws)."""

    iterations: int
    """Number of iterations requested."""

    successful_runs: int = 0
    """Number of successful executions."""

    failed_runs: int = 0
    """Number of failed executions."""

    executions: List[ExecutionResult] = field(default_factory=list)
    """List of all execution results."""

    test_input: Optional[Dict[str, Any]] = None
    """Input used for testing."""

    test_output: Optional[Dict[str, Any]] = None
    """Expected output (if provided)."""

    comparison_results: Optional[Dict[str, Any]] = None
    """Results of comparing output to expected."""

    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    """When validation started."""

    end_time: Optional[str] = None
    """When validation ended."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code_path": self.code_path,
            "runtime": self.runtime,
            "iterations": self.iterations,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "executions": [e.to_dict() for e in self.executions],
            "test_input": self.test_input,
            "test_output": self.test_output,
            "comparison_results": self.comparison_results,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class BaseRunner(ABC):
    """
    Abstract base class for function runners.

    Subclasses must implement:
    - build(): Prepare the function for execution
    - execute(): Run the function with given input
    - cleanup(): Clean up any resources
    """

    def __init__(self, code_path: str, inject_saaf: bool = True):
        """
        Initialize the runner.

        Args:
            code_path: Path to the Python code file.
            inject_saaf: Whether to inject SAAF instrumentation.
        """
        self.code_path = code_path
        self.inject_saaf = inject_saaf
        self._built = False

    @abstractmethod
    def build(self) -> bool:
        """
        Build/prepare the function for execution.

        Returns:
            True if build successful, False otherwise.
        """
        pass

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the function with given input.

        Args:
            input_data: Input dictionary to pass to the function.

        Returns:
            ExecutionResult with output or error.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources created during build/execution."""
        pass

    def run(
        self,
        input_data: Dict[str, Any],
        iterations: int = 1,
        max_total_seconds: float | None = None,
    ) -> List[ExecutionResult]:
        """
        Run the function multiple times.

        Args:
            input_data: Input dictionary to pass to the function.
            iterations: Number of times to execute.
            max_total_seconds: Optional budget in seconds; stop early if exceeded.

        Returns:
            List of ExecutionResults.
        """
        import time

        if not self._built:
            if not self.build():
                return [
                    ExecutionResult(success=False, error="Build failed", iteration=0)
                ]

        results = []
        start = time.monotonic()
        for i in range(iterations):
            if (
                max_total_seconds is not None
                and (time.monotonic() - start) >= max_total_seconds
            ):
                break
            result = self.execute(input_data)
            result.iteration = i
            results.append(result)

        return results

    @property
    def runtime_name(self) -> str:
        """Return the name of this runtime."""
        return self.__class__.__name__.replace("Runner", "").lower()
