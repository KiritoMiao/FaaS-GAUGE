"""
AWS Lambda runner for executing functions on AWS.
"""

import os
import sys
import json
import time
import zipfile
import tempfile
import hashlib
import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseRunner, ExecutionResult
from ..utils import MAX_OUTPUT_BYTES, TOO_LARGE_OUTPUT_FLAG, sanitize_large_output


class AWSRunner(BaseRunner):
    """
    Execute functions on AWS Lambda.

    This runner:
    1. Reads the source code
    2. Injects SAAF instrumentation if enabled
    3. Packages and deploys to AWS Lambda
    4. Invokes the function
    """

    def __init__(
        self,
        code_path: str,
        inject_saaf: bool = True,
        handler_name: Optional[str] = None,
        aws_config: Optional[Dict[str, str]] = None,
        function_name: Optional[str] = None,
        memory_size: int = 1769,
        timeout: int = 30,
        region: str = "us-west-2",
        reuse_deployed_function: bool = True,
    ):
        """
        Initialize the AWS Lambda runner.

        Args:
            code_path: Path to the Python code file.
            inject_saaf: Whether to inject SAAF instrumentation.
            handler_name: Name of the handler function. If None, auto-detects.
            aws_config: Dict with 'profile' and/or 'region' for AWS configuration.
            function_name: Name for the Lambda function. Auto-generated if None.
            memory_size: Memory allocation in MB (128-10240).
            timeout: Timeout in seconds (1-900).
            region: AWS region for Lambda deployment.
            reuse_deployed_function: Reuse existing deployed function when code/config match.
        """
        super().__init__(code_path, inject_saaf)
        self.handler_name = handler_name
        self.aws_config = aws_config or {}
        self.memory_size = memory_size
        self.timeout = timeout
        self.region = region
        self.reuse_deployed_function = reuse_deployed_function

        # Generate function name from code path if not provided
        if function_name is None:
            code_hash = hashlib.md5(code_path.encode()).hexdigest()[:8]
            base_name = Path(code_path).stem
            self.function_name = f"fv-{base_name}-{code_hash}"[:64]
        else:
            self.function_name = function_name

        self._lambda_client: Any = None
        self._iam_client: Any = None
        self._logs_client: Any = None
        self._sts_client: Any = None
        self._aws_account_id: str | None = None
        self._function_arn: str | None = None
        self._role_arn: str | None = None
        self._temp_dir: str | None = None
        self._zip_path: str | None = None

    def _load_login_cache_credentials(self, profile: str) -> Optional[Dict[str, str]]:
        """
        Load credentials from AWS CLI login cache.

        The 'aws login' command stores session credentials in ~/.aws/login/cache/
        """
        import os
        import json
        import glob
        from datetime import datetime

        cache_dir = os.path.expanduser("~/.aws/login/cache")
        if not os.path.isdir(cache_dir):
            return None

        # Find cache files
        cache_files = glob.glob(os.path.join(cache_dir, "*.json"))

        for cache_file in cache_files:
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)

                access_token = data.get("accessToken", {})
                if not access_token:
                    continue

                # Check expiration
                expires_at = access_token.get("expiresAt")
                if expires_at:
                    exp_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    if exp_time < datetime.now(exp_time.tzinfo):
                        continue  # Token expired

                # Return credentials
                return {
                    "aws_access_key_id": access_token.get("accessKeyId"),
                    "aws_secret_access_key": access_token.get("secretAccessKey"),
                    "aws_session_token": access_token.get("sessionToken"),
                }
            except Exception:
                continue

        return None

    def _init_clients(self) -> None:
        """Initialize AWS clients using profile-based authentication."""
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Lambda runner. Install with: pip install boto3"
            )

        session = None
        profile = self.aws_config.get("profile")
        region = self.aws_config.get("region", self.region)

        # Try loading from AWS login cache first
        if profile:
            cache_creds = self._load_login_cache_credentials(profile)
            if cache_creds:
                try:
                    session = boto3.Session(
                        aws_access_key_id=cache_creds["aws_access_key_id"],
                        aws_secret_access_key=cache_creds["aws_secret_access_key"],
                        aws_session_token=cache_creds["aws_session_token"],
                        region_name=region,
                    )
                    # Test credentials
                    sts = session.client("sts")
                    identity = sts.get_caller_identity()
                    self._aws_account_id = identity["Account"]
                except Exception:
                    session = None  # Fall back to profile-based auth

        # Fall back to profile-based authentication
        if session is None:
            session_kwargs = {}
            if profile:
                session_kwargs["profile_name"] = profile
            session_kwargs["region_name"] = region

            try:
                session = boto3.Session(**session_kwargs)
                sts = session.client("sts")
                identity = sts.get_caller_identity()
                self._aws_account_id = identity["Account"]
            except Exception as e:
                # If configured profile is missing locally, retry with default credential chain.
                if (
                    profile
                    and ("profile" in str(e).lower())
                    and ("could not be found" in str(e).lower())
                ):
                    try:
                        session = boto3.Session(region_name=region)
                        sts = session.client("sts")
                        identity = sts.get_caller_identity()
                        self._aws_account_id = identity["Account"]
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"AWS credential validation failed for configured profile '{profile}' ({e}) "
                            f"and default credential chain ({fallback_error}).\n"
                            "Please authenticate with AWS CLI or configure credentials:\n"
                            f"  - aws login --profile {profile}\n"
                            "  - or export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN"
                        )
                else:
                    raise RuntimeError(
                        f"AWS credential validation failed for profile '{profile or 'default'}': {e}\n"
                        "Please check your AWS profile configuration:\n"
                        f"  - Run 'aws login --profile {profile}' to authenticate\n"
                        f"  - Or run 'aws configure --profile {profile}' to set up static credentials"
                    )

        client_config = Config(
            connect_timeout=10,
            read_timeout=max(int(self.timeout) + 30, 120),
            retries={"max_attempts": 2, "mode": "standard"},
        )
        self._lambda_client = session.client("lambda", config=client_config)
        self._iam_client = session.client("iam", config=client_config)
        self._logs_client = session.client("logs", config=client_config)
        self._sts_client = session.client("sts", config=client_config)

    def _read_code(self) -> str:
        """Read the source code from file."""
        with open(self.code_path, "r") as f:
            return f.read()

    def _detect_handler(self, code: str) -> str:
        """Detect the handler function name from code."""
        import re

        # Common handler patterns
        common_handlers = [
            "handler",
            "yourFunction",
            "lambda_handler",
            "main",
            "function_handler",
            "my_handler",
        ]

        for handler_name in common_handlers:
            pattern = rf"def\s+({handler_name})\s*\("
            match = re.search(pattern, code)
            if match:
                return match.group(1)

        # Fallback: find function with two parameters (with optional type annotations)
        pattern = r"def\s+(\w+)\s*\(\s*\w+\s*(?::\s*[^,)]+)?\s*,\s*\w*"
        match = re.search(pattern, code)
        if match:
            return match.group(1)

        raise ValueError("Could not detect handler function. Please specify handler_name.")

    def _inject_saaf(self, code: str, handler_name: str) -> str:
        """
        Inject SAAF instrumentation into the code.

        SAAF pattern:
        1. Initialize Inspector and call inspectAll() at the start
        2. Execute original function logic
        3. Use addAttribute() to add result keys for validation
        4. Call inspectAllDeltas() before returning
        5. Return inspector.finish()
        """
        saaf_path = Path(__file__).parent.parent / "saaf.py"

        with open(saaf_path, "r") as f:
            saaf_code = f.read()

        # Keep Lambda response payload under safe bounds.
        max_output_bytes = MAX_OUTPUT_BYTES
        too_large_flag = TOO_LARGE_OUTPUT_FLAG
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

def _saaf_json_size(value):
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="replace"))

def _saaf_sanitize_output(value, max_bytes={max_output_bytes}, flag="{too_large_flag}"):
    if value is None:
        return value, False, []

    if _saaf_json_size(value) <= max_bytes:
        return value, False, []

    if isinstance(value, dict):
        sanitized = {{}}
        replaced_keys = []
        for key, item in value.items():
            if _saaf_json_size(item) > max_bytes:
                sanitized[key] = flag
                replaced_keys.append(str(key))
            else:
                sanitized[key] = item

        if (not replaced_keys) or _saaf_json_size(sanitized) > max_bytes:
            return {{"__validator_output__": flag}}, True, ["__all__"]
        return sanitized, True, replaced_keys

    if isinstance(value, list):
        sanitized = []
        replaced = False
        for item in value:
            if _saaf_json_size(item) > max_bytes:
                sanitized.append(flag)
                replaced = True
            else:
                sanitized.append(item)

        if (not replaced) or _saaf_json_size(sanitized) > max_bytes:
            return flag, True, ["__all__"]
        return sanitized, True, ["__list__"]

    return flag, True, ["__value__"]

def {handler_name}(request, context=None):
    """SAAF-wrapped handler function."""
    request = _saaf_materialize_request(request)

    # Initialize SAAF at the start
    inspector = Inspector()
    inspector.inspectAll()
    
    try:
        # Call original function
        result = _original_{handler_name}(request, context)

        sanitized_result, output_too_large, too_large_keys = _saaf_sanitize_output(result)

        # Use addAttribute to add result for validation.
        if isinstance(sanitized_result, dict):
            for key, value in sanitized_result.items():
                inspector.addAttribute(key, value)
        else:
            inspector.addAttribute("result", sanitized_result)

        inspector.addAttribute("success", True)
        inspector.addAttribute("__output_too_large__", output_too_large)
        if too_large_keys:
            inspector.addAttribute("__output_too_large_keys__", too_large_keys)
        
    except Exception as e:
        inspector.addAttribute("error", str(e))
        inspector.addAttribute("errorType", type(e).__name__)
        inspector.addAttribute("success", False)
    
    # Finalize SAAF at return
    inspector.inspectAllDeltas()
    return inspector.finish()

# === SAAF INJECTION END ===
'''

        if "Inspector()" in code or "# === SAAF INJECTION" in code:
            return code

        return code + "\n" + wrapper

    def _get_or_create_role(self) -> str:
        """Get or create IAM role for Lambda execution."""
        role_name = "function-validator-lambda-role"

        try:
            response = self._iam_client.get_role(RoleName=role_name)
            return response["Role"]["Arn"]  # type: ignore[no-any-return]
        except self._iam_client.exceptions.NoSuchEntityException:
            pass

        # Create role
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        response = self._iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for function-validator Lambda executions",
        )

        # Attach basic execution policy
        self._iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        # Wait for role to propagate
        time.sleep(10)

        return response["Role"]["Arn"]  # type: ignore[no-any-return]

    def _install_dependencies(self, packages: List[str], target_dir: str) -> bool:
        """
        Install pip packages into target directory.

        Args:
            packages: List of package names to install.
            target_dir: Directory to install packages into.

        Returns:
            True if installation successful.
        """
        import subprocess

        if not packages:
            return True

        try:
            # Install packages to target directory
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                target_dir,
                "--quiet",
                "--no-cache-dir",
                "--platform",
                "manylinux2014_x86_64",
                "--only-binary=:all:",
                "--python-version",
                "3.11",
            ] + packages

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                # Try without platform restrictions (for pure Python packages)
                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    target_dir,
                    "--quiet",
                    "--no-cache-dir",
                ] + packages

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            return result.returncode == 0

        except Exception as e:
            print(f"Warning: Failed to install dependencies: {e}")
            return False

    def _create_deployment_package(self, code: str, dependencies: List[str] | None = None) -> str:
        """
        Create a ZIP deployment package with dependencies.

        Args:
            code: Lambda function code.
            dependencies: List of pip packages to include.

        Returns:
            Path to the ZIP file.
        """
        self._temp_dir = tempfile.mkdtemp()
        assert self._temp_dir is not None
        package_dir = os.path.join(self._temp_dir, "package")
        os.makedirs(package_dir)

        # Write lambda function
        lambda_file = os.path.join(package_dir, "lambda_function.py")
        with open(lambda_file, "w") as f:
            f.write(code)

        # Install dependencies if any
        if dependencies:
            print(f"Installing dependencies: {', '.join(dependencies)}")
            self._install_dependencies(dependencies, package_dir)

        # Create ZIP with all files
        self._zip_path = os.path.join(self._temp_dir, "deployment.zip")
        assert self._zip_path is not None
        with zipfile.ZipFile(self._zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(package_dir):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]

                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_dir)
                    zf.write(file_path, arcname)

        # Check ZIP size (Lambda limit is 50MB zipped, 250MB unzipped)
        zip_size = os.path.getsize(self._zip_path)
        if zip_size > 50 * 1024 * 1024:
            print(f"Warning: Deployment package is {zip_size / 1024 / 1024:.1f}MB (limit: 50MB)")

        assert self._zip_path is not None
        return self._zip_path

    def _wait_for_function_ready(self) -> None:
        """Wait for Lambda function to be ready for updates."""
        import time

        max_attempts = 30
        for _ in range(max_attempts):
            try:
                response = self._lambda_client.get_function(FunctionName=self.function_name)
                state = response["Configuration"].get("State", "Active")
                last_update = response["Configuration"].get("LastUpdateStatus", "Successful")

                if state == "Active" and last_update in ("Successful", None):
                    return

                time.sleep(1)
            except Exception:
                return  # Function might not exist yet

    def _deploy_function(self, zip_path: str) -> str:
        """Deploy or update Lambda function."""
        with open(zip_path, "rb") as f:
            zip_content = f.read()

        handler_module = f"lambda_function.{self.handler_name}"
        local_code_sha = base64.b64encode(hashlib.sha256(zip_content).digest()).decode("utf-8")

        try:
            existing = self._lambda_client.get_function(FunctionName=self.function_name)
            config = existing.get("Configuration", {})
            function_arn = config.get("FunctionArn")

            if config.get("CodeSha256") == local_code_sha:
                code_matches = True
            else:
                code_matches = False

            config_matches = (
                config.get("Handler") == handler_module
                and int(config.get("MemorySize", 0)) == int(self.memory_size)
                and int(config.get("Timeout", 0)) == int(self.timeout)
            )

            if self.reuse_deployed_function and code_matches and config_matches:
                return function_arn  # type: ignore[no-any-return]

            self._wait_for_function_ready()

            if (not code_matches) or (not self.reuse_deployed_function):
                response = self._lambda_client.update_function_code(
                    FunctionName=self.function_name, ZipFile=zip_content
                )
                function_arn = response.get("FunctionArn", function_arn)
                self._wait_for_function_ready()

            if (not config_matches) or (not self.reuse_deployed_function):
                self._lambda_client.update_function_configuration(
                    FunctionName=self.function_name,
                    MemorySize=self.memory_size,
                    Timeout=self.timeout,
                    Handler=handler_module,
                )
                self._wait_for_function_ready()

            return function_arn  # type: ignore[no-any-return]

        except self._lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            response = self._lambda_client.create_function(
                FunctionName=self.function_name,
                Runtime="python3.11",
                Role=self._role_arn,
                Handler=handler_module,
                Code={"ZipFile": zip_content},
                MemorySize=self.memory_size,
                Timeout=self.timeout,
                Description=f"Function Validator: {self.code_path}",
            )

            # Wait for function to be active
            waiter = self._lambda_client.get_waiter("function_active")
            waiter.wait(FunctionName=self.function_name)

            return response["FunctionArn"]  # type: ignore[no-any-return]

    def _decode_log_tail(self, response: Dict[str, Any]) -> str:
        """Decode invoke tail logs (base64) when present."""
        encoded = response.get("LogResult")
        if not encoded:
            return ""
        try:
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _extract_request_id(self, log_text: str) -> Optional[str]:
        """Extract Lambda request id from invoke tail logs."""
        for pattern in (
            r"REPORT RequestId:\s*([a-fA-F0-9-]+)",
            r"START RequestId:\s*([a-fA-F0-9-]+)",
        ):
            match = re.search(pattern, log_text)
            if match:
                return match.group(1)
        return None

    def _extract_max_memory_used_mb(self, log_text: str) -> Optional[float]:
        """Extract max memory usage from REPORT lines in logs."""
        matches = re.findall(r"Max Memory Used:\s*([0-9]+)\s*MB", log_text)
        if not matches:
            return None
        try:
            return max(float(m) for m in matches)
        except Exception:
            return None

    def _fetch_cloudwatch_max_memory_used_mb(
        self, request_id: Optional[str], start_ms: int, end_ms: int
    ) -> Optional[float]:
        """
        Query CloudWatch logs for REPORT lines around one invocation window.
        """
        if self._logs_client is None:
            return None

        log_group = f"/aws/lambda/{self.function_name}"
        start_ms = max(0, int(start_ms) - 120000)
        end_ms = int(end_ms) + 120000
        memory_values: List[float] = []

        if request_id:
            filter_pattern = f'"REPORT" "{request_id}"'
        else:
            filter_pattern = '"REPORT"'

        try:
            paginator = self._logs_client.get_paginator("filter_log_events")
            for page in paginator.paginate(
                logGroupName=log_group,
                startTime=start_ms,
                endTime=end_ms,
                filterPattern=filter_pattern,
            ):
                for event in page.get("events", []):
                    message = event.get("message", "")
                    if request_id and request_id not in message:
                        continue
                    mem = self._extract_max_memory_used_mb(message)
                    if mem is not None:
                        memory_values.append(mem)
        except Exception:
            return None

        if not memory_values:
            return None
        return max(memory_values)

    def build(self) -> bool:
        """
        Build and deploy the function to AWS Lambda.

        Automatically detects and packages third-party dependencies.

        Returns:
            True if build successful.
        """
        try:
            self._init_clients()

            # Read and prepare code
            original_code = self._read_code()

            if self.handler_name is None:
                self.handler_name = self._detect_handler(original_code)

            if self.inject_saaf:
                modified_code = self._inject_saaf(original_code, self.handler_name)
            else:
                modified_code = original_code

            # Detect third-party dependencies
            from ..dependency import detect_dependencies

            code_dir = os.path.dirname(os.path.abspath(self.code_path))
            dependencies, import_names = detect_dependencies(original_code, code_dir)

            if dependencies:
                print(f"Detected dependencies: {', '.join(dependencies)}")

            # Get/create IAM role
            self._role_arn = self._get_or_create_role()
            assert self._role_arn is not None

            # Create deployment package with dependencies
            zip_path = self._create_deployment_package(modified_code, dependencies)

            # Deploy function
            self._function_arn = self._deploy_function(zip_path)

            self._built = True
            return True

        except Exception as e:
            self._built = False
            self._build_error = str(e)
            import traceback

            traceback.print_exc()
            return False

    def execute(self, input_data: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the function on AWS Lambda.

        Args:
            input_data: Input dictionary to pass to the function.

        Returns:
            ExecutionResult with output or error.
        """
        if not self._built:
            return ExecutionResult(
                success=False,
                error=f"Function not built. Error: {getattr(self, '_build_error', 'Unknown')}",
            )

        start_time = time.time()
        invoke_start_ms = int(start_time * 1000)

        try:
            response = self._lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                LogType="Tail",
                Payload=json.dumps(input_data),
            )

            execution_time = (time.time() - start_time) * 1000
            invoke_end_ms = int(time.time() * 1000)

            # Read response
            payload = response["Payload"].read().decode("utf-8")
            try:
                result = json.loads(payload)
            except json.JSONDecodeError:
                result = {"rawPayload": payload}

            log_tail = self._decode_log_tail(response)
            request_id = self._extract_request_id(log_tail)
            max_memory_used_mb = self._extract_max_memory_used_mb(log_tail)
            if max_memory_used_mb is None:
                max_memory_used_mb = self._fetch_cloudwatch_max_memory_used_mb(
                    request_id=request_id,
                    start_ms=invoke_start_ms,
                    end_ms=invoke_end_ms,
                )

            sanitized_result, output_too_large = sanitize_large_output(
                result,
                max_bytes=MAX_OUTPUT_BYTES,
                flag=TOO_LARGE_OUTPUT_FLAG,
            )
            if isinstance(result, dict) and result.get("__output_too_large__"):
                output_too_large = True

            # Check for Lambda errors
            if "FunctionError" in response:
                if isinstance(result, dict):
                    error_message = result.get("errorMessage", str(result))
                else:
                    error_message = str(result)
                return ExecutionResult(
                    success=False,
                    error=error_message,
                    output=sanitized_result,
                    execution_time_ms=execution_time,
                    output_too_large=output_too_large,
                    max_memory_used_mb=max_memory_used_mb,
                    cloudwatch_request_id=request_id,
                )

            return ExecutionResult(
                success=True,
                output=sanitized_result,
                execution_time_ms=execution_time,
                output_too_large=output_too_large,
                max_memory_used_mb=max_memory_used_mb,
                cloudwatch_request_id=request_id,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=execution_time,
            )

    def cleanup(self) -> None:
        """Clean up temporary files and optionally the Lambda function."""
        import shutil

        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

        self._built = False

    def delete_function(self) -> bool:
        """
        Delete the Lambda function from AWS.

        Returns:
            True if deletion successful.
        """
        if self._lambda_client is None:
            self._init_clients()

        try:
            self._lambda_client.delete_function(FunctionName=self.function_name)
            return True
        except Exception:
            return False

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
