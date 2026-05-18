"""
Local runner for executing functions directly in Python.
"""

import os
import sys
import time
import json
import signal
import tempfile
import traceback
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional
import importlib.util

from .base import BaseRunner, ExecutionResult
from ..utils import sanitize_large_output


class TimeoutError(Exception):
    """Raised when function execution times out."""

    pass


class LocalRunner(BaseRunner):
    """
    Execute functions locally using Python's import system.

    This runner:
    1. Reads the source code
    2. Injects SAAF instrumentation if enabled
    3. Executes the function in a subprocess or via import
    """

    def __init__(
        self,
        code_path: str,
        inject_saaf: bool = True,
        handler_name: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize the local runner.

        Args:
            code_path: Path to the Python code file.
            inject_saaf: Whether to inject SAAF instrumentation.
            handler_name: Name of the handler function. If None, auto-detects.
            timeout: Maximum execution time in seconds (default: 300).
        """
        super().__init__(code_path, inject_saaf)
        self.handler_name = handler_name
        self.timeout = timeout
        self._module: ModuleType | None = None
        self._temp_file: str | None = None
        self._original_code: str | None = None
        self._modified_code: str | None = None

    def _read_code(self) -> str:
        """Read the source code from file."""
        with open(self.code_path, "r") as f:
            return f.read()

    def _detect_handler(self, code: str) -> str:
        """
        Detect the handler function name from code.

        Args:
            code: Python source code.

        Returns:
            Handler function name.
        """
        import re

        # Common handler patterns (with optional type annotations)
        common_handlers = [
            "handler",
            "yourFunction",
            "lambda_handler",
            "main",
            "function_handler",
            "my_handler",
        ]

        for handler_name in common_handlers:
            # Match with or without type annotations
            pattern = rf"def\s+({handler_name})\s*\("
            match = re.search(pattern, code)
            if match:
                return match.group(1)

        # Fallback: find function with two parameters (with optional type annotations)
        # Matches: def func(param1, param2) or def func(param1: Type, param2: Type)
        pattern = r"def\s+(\w+)\s*\(\s*\w+\s*(?::\s*[^,)]+)?\s*,\s*\w*"
        match = re.search(pattern, code)
        if match:
            return match.group(1)

        raise ValueError(
            "Could not detect handler function. Please specify handler_name."
        )

    def _inject_saaf(self, code: str, handler_name: str) -> str:
        """
        Inject SAAF instrumentation into the code.

        SAAF pattern:
        1. Initialize Inspector and call inspectAll() at the start
        2. Execute original function logic
        3. Use addAttribute() to add result keys for validation
        4. Call inspectAllDeltas() before returning
        5. Return inspector.finish()

        Args:
            code: Original source code.
            handler_name: Name of the handler function to wrap.

        Returns:
            Modified code with SAAF injection.
        """
        # Get the path to SAAF.py in the same package
        saaf_path = Path(__file__).parent.parent / "saaf.py"

        # Read SAAF code
        with open(saaf_path, "r") as f:
            saaf_code = f.read()

        # Create wrapper code following the SAAF pattern
        wrapper = f'''
# === SAAF INJECTION START ===
{saaf_code}

# Store original handler
_original_{handler_name} = {handler_name}

def _saaf_build_median_of_two_array(params):
    import random

    safe_params = params if isinstance(params, dict) else dict()
    nums1_len = max(0, int(safe_params.get("nums1_len", 0)))
    nums2_len = max(0, int(safe_params.get("nums2_len", 0)))
    seed = int(safe_params.get("seed", 42))
    min_value = int(safe_params.get("min_value", 0))
    max_value = int(safe_params.get("max_value", 10_000_000))
    if max_value < min_value:
        max_value = min_value

    rng = random.Random(seed)
    nums1 = sorted(rng.randint(min_value, max_value) for _ in range(nums1_len))
    nums2 = sorted(rng.randint(min_value, max_value) for _ in range(nums2_len))
    return dict(nums1=nums1, nums2=nums2)

def _saaf_materialize_request(request):
    if not isinstance(request, dict):
        return request

    descriptor = request.get("__defer_generate__")
    if not isinstance(descriptor, dict):
        return request

    generator_name = str(descriptor.get("generator") or "").strip().lower()
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else dict()
    if generator_name == "median_of_two_array":
        return _saaf_build_median_of_two_array(params)

    return request

def _saaf_unwrap_lambda_response(result):
    """Unwrap Lambda HTTP-style response to get the actual payload."""
    import json as _json
    if not isinstance(result, dict):
        return result
    # Detect Lambda proxy response pattern
    if "statusCode" in result and "body" in result:
        body = result["body"]
        if isinstance(body, str):
            try:
                return _json.loads(body)
            except (_json.JSONDecodeError, ValueError):
                return result
        elif isinstance(body, dict):
            return body
    return result

def {handler_name}(request, context=None):
    """SAAF-wrapped handler function."""
    request = _saaf_materialize_request(request)

    # Initialize SAAF at the start (platform-safe)
    inspector = Inspector()
    import platform as _platform
    if _platform.system() == 'Linux':
        inspector.inspectAll()
    else:
        # macOS/Windows: skip /proc-dependent methods
        inspector.inspectContainer()
        inspector.inspectPlatform()
        inspector.addTimeStamp("frameworkRuntime")
    
    try:
        # Call original function
        result = _original_{handler_name}(request, context)
        
        # Unwrap Lambda HTTP responses to get actual payload
        result = _saaf_unwrap_lambda_response(result)
        
        # Use addAttribute to add result for validation
        if isinstance(result, dict):
            for key, value in result.items():
                inspector.addAttribute(key, value)
        else:
            inspector.addAttribute("result", result)
        
        inspector.addAttribute("success", True)
        
    except (KeyboardInterrupt, SystemExit):
        # Re-raise critical exceptions
        raise
    except BaseException as e:
        # Check if this is a timeout error (re-raise it)
        if 'timeout' in type(e).__name__.lower() or 'timeout' in str(e).lower():
            raise
        inspector.addAttribute("error", str(e))
        inspector.addAttribute("errorType", type(e).__name__)
        inspector.addAttribute("success", False)
    
    # Finalize SAAF at return (platform-safe)
    if _platform.system() == 'Linux':
        inspector.inspectAllDeltas()
    else:
        if "frameworkRuntime" in inspector._Inspector__attributes:
            inspector.addTimeStamp(
                "userRuntime",
                inspector._Inspector__startTime + inspector._Inspector__attributes["frameworkRuntime"]
            )
        inspector.addTimeStamp("frameworkRuntimeDeltas")
    return inspector.finish()

# === SAAF INJECTION END ===
'''

        # Check if the code already has SAAF
        if "Inspector()" in code or "# === SAAF INJECTION" in code:
            # Already has SAAF, return as-is
            return code

        # Append wrapper after the original code
        return code + "\n" + wrapper

    def build(self) -> bool:
        """
        Build the function by preparing the code with SAAF injection.

        Detects and warns about missing third-party dependencies.

        Returns:
            True if build successful.
        """
        try:
            self._original_code = self._read_code()

            # Detect handler name if not provided
            if self.handler_name is None:
                self.handler_name = self._detect_handler(self._original_code)

            # Detect third-party dependencies
            code_dir = os.path.dirname(os.path.abspath(self.code_path))
            from ..dependency import detect_dependencies, get_missing_packages

            dependencies, import_names = detect_dependencies(
                self._original_code, code_dir
            )
            if dependencies:
                missing = get_missing_packages(dependencies)
                if missing:
                    print(f"Warning: Missing packages detected: {', '.join(missing)}")
                    print(f"Install with: pip install {' '.join(missing)}")

            if self.inject_saaf:
                assert self.handler_name is not None
                self._modified_code = self._inject_saaf(
                    self._original_code, self.handler_name
                )
            else:
                self._modified_code = self._original_code

            # Create a temporary file for the modified code
            fd, self._temp_file = tempfile.mkstemp(suffix=".py")
            assert self._modified_code is not None
            with os.fdopen(fd, "w") as f:
                f.write(self._modified_code)

            # Import the module
            spec = importlib.util.spec_from_file_location(
                "temp_module", self._temp_file
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load module spec from {self._temp_file}")
            self._module = importlib.util.module_from_spec(spec)

            # Add the original code's directory to path for imports
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)

            spec.loader.exec_module(self._module)

            self._built = True
            return True

        except Exception as e:
            self._built = False
            self._build_error = str(e)
            # Don't print traceback - just store the error
            return False

    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise TimeoutError(f"Function execution timed out after {self.timeout} seconds")

    def execute(self, input_data: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the function with given input.

        Args:
            input_data: Input dictionary to pass to the function.

        Returns:
            ExecutionResult with output or error.
        """
        if not self._built or self._module is None:
            return ExecutionResult(
                success=False,
                error=f"Module not built. Error: {getattr(self, '_build_error', 'Unknown')}",
            )

        start_time = time.time()
        old_handler = None

        try:
            # Set up timeout using signal (Unix only)
            if hasattr(signal, "SIGALRM") and self.timeout > 0:
                old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.timeout)

            assert self.handler_name is not None
            handler = getattr(self._module, self.handler_name)
            result = handler(input_data, {})
            sanitized_result, output_too_large = sanitize_large_output(result)

            # Cancel the alarm
            if hasattr(signal, "SIGALRM") and self.timeout > 0:
                signal.alarm(0)

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=True,
                output=sanitized_result,
                execution_time_ms=execution_time,
                output_too_large=output_too_large,
            )

        except TimeoutError as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False, error=str(e), execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=execution_time,
            )

        finally:
            # Restore old signal handler and cancel alarm
            if hasattr(signal, "SIGALRM") and self.timeout > 0:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.remove(self._temp_file)
            except Exception:
                pass
        self._module = None
        self._built = False

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
