"""
Main FunctionValidator class - the primary interface for the package.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from faas_gauge.config import load_credentials
from faas_gauge.validator.runners import AWSRunner, LocalRunner
from faas_gauge.validator.runners.base import (
    BaseRunner,
    ExecutionResult,
    ValidationResult,
)
from faas_gauge.validator.utils import compare_results, load_json_input, save_results


class FunctionValidator:
    """
    Universal Function Validator with SAAF injection.

    Provides a unified interface for building, executing, and validating
    serverless functions with SAAF (Serverless Application Analytics Framework)
    instrumentation.

    Example usage:
        >>> validator = FunctionValidator(
        ...     code_path="my_function.py",
        ...     runtime="local",
        ...     test_input={"key": "value"}
        ... )
        >>> result = validator.run(iterations=5, save_result=True)
        >>> print(f"Success rate: {result.successful_runs}/{result.iterations}")
    """

    SUPPORTED_RUNTIMES = ["local", "aws"]

    def __init__(
        self,
        code_path: str,
        runtime: str = "local",
        test_input: Optional[Union[str, Dict, Path]] = None,
        test_output: Optional[Union[str, Dict, Path]] = None,
        inject_saaf: bool = True,
        handler_name: Optional[str] = None,
        config_path: Optional[str] = None,
        aws_config: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        aws_memory_size: int = 1769,
        reuse_deployed_function: bool = True,
    ):
        """
        Initialize the FunctionValidator.

        Args:
            code_path: Path to the Python code file to validate.
            runtime: Execution runtime - "local" or "aws".
            test_input: Input for the function test. Can be:
                - A dictionary
                - A JSON string
                - A path to a JSON file
            test_output: Optional expected output. Same format as test_input.
                SAAF attributes are excluded from comparison.
            inject_saaf: Whether to inject SAAF instrumentation (default True).
            handler_name: Name of the handler function. Auto-detected if None.
            config_path: Path to credentials JSON config. Auto-discovered if None.
            aws_config: Optional AWS configuration override.
            timeout: Maximum execution time in seconds (default: 300).
            aws_memory_size: AWS Lambda memory size in MB (default: 1769).
            reuse_deployed_function: Reuse existing AWS Lambda when code/config are unchanged.
        """
        self.code_path = os.path.abspath(code_path)

        if not os.path.isfile(self.code_path):
            raise FileNotFoundError(f"Code file not found: {self.code_path}")

        self.runtime = runtime.lower()
        if self.runtime not in self.SUPPORTED_RUNTIMES:
            raise ValueError(
                f"Unsupported runtime: {runtime}. "
                f"Supported: {', '.join(self.SUPPORTED_RUNTIMES)}"
            )

        self.inject_saaf = inject_saaf
        self.handler_name = handler_name
        self.config_path = config_path
        self.aws_config = aws_config
        self.timeout = timeout
        self.aws_memory_size = aws_memory_size
        self.reuse_deployed_function = reuse_deployed_function

        # Load test input/output
        self.test_input = self._load_input(test_input) if test_input else {}
        self.test_output = self._load_input(test_output) if test_output else None

        # Initialize runner
        self._runner: BaseRunner | None = None
        self._secrets: Dict[str, Any] | None = None

    def _load_input(self, data: Union[str, Dict, Path]) -> Dict[str, Any]:
        """Load JSON input from various sources."""
        return load_json_input(data)

    def _get_aws_config(self) -> Dict[str, str]:
        """Get AWS configuration (profile and region) from credentials config."""
        if self.aws_config:
            return self.aws_config

        if self._secrets is None:
            try:
                self._secrets = load_credentials(self.config_path)
            except FileNotFoundError:
                self._secrets = {}

        aws_section = self._secrets.get("aws", {})

        return {
            "profile": aws_section.get("profile"),
            "region": aws_section.get("region", "us-west-2"),
        }

    def _create_runner(self):
        """Create the appropriate runner based on runtime."""
        if self.runtime == "local":
            return LocalRunner(
                code_path=self.code_path,
                inject_saaf=self.inject_saaf,
                handler_name=self.handler_name,
                timeout=self.timeout,
            )
        elif self.runtime == "aws":
            return AWSRunner(
                code_path=self.code_path,
                inject_saaf=self.inject_saaf,
                handler_name=self.handler_name,
                aws_config=self._get_aws_config(),
                memory_size=self.aws_memory_size,
                timeout=self.timeout,
                reuse_deployed_function=self.reuse_deployed_function,
            )
        else:
            raise ValueError(f"Unknown runtime: {self.runtime}")

    def build(self) -> bool:
        """
        Build/prepare the function for execution.

        Returns:
            True if build successful.
        """
        self._runner = self._create_runner()
        return bool(self._runner.build())

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute the function once with given input.

        Args:
            input_data: Input to pass to function. Uses test_input if None.

        Returns:
            ExecutionResult with output and metrics.
        """
        if self._runner is None:
            self.build()
        assert self._runner is not None

        data = input_data if input_data is not None else self.test_input
        return self._runner.execute(data)

    def run(
        self,
        iterations: int = 1,
        input_data: Optional[Dict[str, Any]] = None,
        save_result: bool = False,
        result_filename: Optional[str] = None,
        max_total_seconds: float | None = None,
    ) -> ValidationResult:
        """
        Run validation with multiple iterations.

        Args:
            iterations: Number of times to execute the function.
            input_data: Input to pass to function. Uses test_input if None.
            save_result: If True, save results to JSON file in code directory.
            result_filename: Custom filename for results. Auto-generated if None.

        Returns:
            ValidationResult containing all execution results and comparison.
        """
        if self._runner is None:
            if not self.build():
                return ValidationResult(
                    code_path=self.code_path,
                    runtime=self.runtime,
                    iterations=iterations,
                    failed_runs=iterations,
                    test_input=self.test_input,
                    test_output=self.test_output,
                )

        assert self._runner is not None
        data = input_data if input_data is not None else self.test_input

        # Create result object
        result = ValidationResult(
            code_path=self.code_path,
            runtime=self.runtime,
            iterations=iterations,
            test_input=data,
            test_output=self.test_output,
        )

        # Run iterations
        executions = self._runner.run(
            data, iterations, max_total_seconds=max_total_seconds
        )
        result.executions = executions

        # Count successes/failures (based on actual executions, not requested iterations)
        result.iterations = len(executions)
        result.successful_runs = sum(1 for e in executions if e.success)
        result.failed_runs = len(executions) - result.successful_runs

        # Compare with expected output if provided
        if self.test_output and result.successful_runs > 0:
            # Use the first successful result for comparison
            for execution in executions:
                if execution.success and execution.output:
                    result.comparison_results = compare_results(
                        actual=execution.output,
                        expected=self.test_output,
                        ignore_saaf=True,
                    )
                    break

        result.end_time = datetime.now().isoformat()

        # Save results if requested
        if save_result:
            output_path = save_results(
                results=result.to_dict(),
                code_path=self.code_path,
                filename=result_filename,
            )
            print(f"Results saved to: {output_path}")

        return result

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._runner:
            self._runner.cleanup()
            self._runner = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


def validate(
    code_path: str,
    runtime: str = "local",
    test_input: Optional[Union[str, Dict]] = None,
    test_output: Optional[Union[str, Dict]] = None,
    iterations: int = 1,
    save_result: bool = False,
    inject_saaf: bool = True,
    handler_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> ValidationResult:
    """
    Convenience function for quick validation.

    This is a functional interface that creates a FunctionValidator,
    runs validation, and cleans up.

    Args:
        code_path: Path to the Python code file.
        runtime: "local" or "aws".
        test_input: Input for the function.
        test_output: Optional expected output.
        iterations: Number of times to run.
        save_result: Whether to save results to file.
        inject_saaf: Whether to inject SAAF.
        handler_name: Handler function name.
        config_path: Path to credentials file.

    Returns:
        ValidationResult with all execution data.

    Example:
        >>> from faas_gauge.validator import validate
        >>> result = validate(
        ...     code_path="my_func.py",
        ...     runtime="local",
        ...     test_input={"n": 10},
        ...     iterations=5
        ... )
    """
    with FunctionValidator(
        code_path=code_path,
        runtime=runtime,
        test_input=test_input,
        test_output=test_output,
        inject_saaf=inject_saaf,
        handler_name=handler_name,
        config_path=config_path,
    ) as validator:
        run_result: ValidationResult = validator.run(
            iterations=iterations, save_result=save_result
        )
        return run_result
