"""Tests for faas_gauge.validator package."""

from pathlib import Path

from faas_gauge.ai.utils import extract_python_code
from faas_gauge.validator import FunctionValidator
from faas_gauge.validator.dependency import detect_dependencies
from faas_gauge.validator.runners import LocalRunner
from faas_gauge.validator.utils import filter_saaf_attributes
from faas_gauge.validator.validators import (
    get_validator,
    validate_prime_number_generator,
)


def test_local_runner_executes_simple_handler(tmp_path: Path) -> None:
    code_file = tmp_path / "handler.py"
    code_file.write_text(
        "def handler(request, context):\n    return {'result': request.get('x', 0) * 2}\n",
        encoding="utf-8",
    )

    runner = LocalRunner(code_path=str(code_file), inject_saaf=False)
    assert runner.build() is True

    result = runner.execute({"x": 3})
    runner.cleanup()

    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.result == {"result": 6}


def test_function_validator_creation_and_build(tmp_path: Path) -> None:
    code_file = tmp_path / "handler.py"
    code_file.write_text(
        "def handler(request, context):\n    return {'ok': True, 'x': request.get('x', 0)}\n",
        encoding="utf-8",
    )

    validator = FunctionValidator(
        code_path=str(code_file),
        runtime="local",
        test_input={"x": 9},
        inject_saaf=False,
    )
    try:
        assert validator.build() is True
        execution = validator.execute()
        assert execution.success is True
        assert execution.result == {"ok": True, "x": 9}
    finally:
        validator.cleanup()


def test_validators_registry() -> None:
    validator = get_validator("prime-number-generator")
    assert validator is validate_prime_number_generator


def test_extract_python_code_and_filter_saaf_attributes() -> None:
    text = "Some response\n```python\ndef handler(request, context):\n    return {'a': 1}\n```"
    code = extract_python_code(text)
    assert "def handler" in code

    filtered = filter_saaf_attributes({"result": 1, "version": 0.7, "runtime": 10, "success": True})
    assert filtered == {"result": 1}


def test_detect_dependencies() -> None:
    code = "import requests\nimport json\nfrom yaml import safe_load\n"
    packages, imports = detect_dependencies(code, code_dir=".")

    assert "requests" in packages
    assert "pyyaml" in packages
    assert "requests" in imports
    assert "yaml" in imports
    assert "json" not in imports
