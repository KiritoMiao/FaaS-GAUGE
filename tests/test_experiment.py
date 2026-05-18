"""Tests for faas_gauge.experiment modules."""

from unittest.mock import MagicMock, patch

from faas_gauge.ai import AIResponse
from faas_gauge.experiment import Experiment, generate, load_experiment, retry_failed
from faas_gauge.experiment.classify import classify_function_status
from faas_gauge.store import ExperimentStore


def _make_store(data_dir):
    for sub in [
        "config",
        "experiments",
        "validations",
        "questions",
        "test_data",
        "prompts",
    ]:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    return ExperimentStore(data_dir=data_dir)


def _make_response(iteration: int, error: str | None = None, output_tokens: int = 10) -> AIResponse:
    return AIResponse(
        raw_result=f"raw-{iteration}",
        python_code=f"print({iteration})",
        input_tokens=5,
        output_tokens=output_tokens,
        tokens_per_second=20.0,
        cost=0.01,
        rtt_sec=0.5,
        provider="openai",
        model="gpt-4o-mini",
        error=error,
    )


def test_classify_function_status_variants() -> None:
    assert classify_function_status({"tests": []})[0] == "functional-error"

    success = {"tests": [{"name": "default", "status": "passed", "message": "ok"}]}
    assert classify_function_status(success)[0] == "success"

    syntax = {
        "tests": [
            {
                "name": "default",
                "status": "error",
                "message": "SyntaxError: invalid syntax",
            }
        ]
    }
    assert classify_function_status(syntax)[0] == "syntax-error"

    timeout = {
        "tests": [
            {
                "name": "default",
                "status": "error",
                "message": "Function execution timed out",
            }
        ]
    }
    assert classify_function_status(timeout)[0] == "timeout"

    oom = {
        "tests": [
            {
                "name": "default",
                "status": "error",
                "execution_results": [{"error": "Runtime exited with error: signal: killed"}],
            }
        ]
    }
    assert classify_function_status(oom)[0] == "out-of-memory"

    scaling = {
        "tests": [
            {"name": "default", "status": "passed", "message": "ok"},
            {"name": "performance_big", "status": "failed", "message": "too slow"},
        ]
    }
    assert classify_function_status(scaling)[0] == "scaling-bug"


def test_experiment_wrapper_repr_and_iteration_access(tmp_path) -> None:
    store = _make_store(tmp_path / "data")
    meta = {
        "id": "exp_1",
        "question_name": "q",
        "question_content": "content",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "iterations": 1,
    }
    store.create_experiment(meta)
    store.append_iteration("exp_1", {"iteration": 1, "python_code": "print(1)", "error": None})

    exp = Experiment(meta, store)
    assert repr(exp) == "Experiment(exp_1)"
    assert exp["question_name"] == "q"
    assert list(exp.iterations())[0]["iteration"] == 1


def test_load_experiment_reads_from_store(tmp_path) -> None:
    store = _make_store(tmp_path / "data")
    meta = {
        "id": "exp_2",
        "question_name": "sum",
        "question_content": "add two numbers",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "iterations": 1,
        "summary": {"successful_iterations": 1},
    }
    store.create_experiment(meta)

    loaded = load_experiment("exp_2", data_dir=str(tmp_path / "data"))
    assert loaded.id == "exp_2"
    assert loaded.summary == {"successful_iterations": 1}


@patch("faas_gauge.experiment.generate.AIClient")
def test_generate_with_mocked_ai_client(mock_ai_client: MagicMock, tmp_path) -> None:
    store = _make_store(tmp_path / "data")
    del store

    client = MagicMock()
    client.ask.side_effect = [_make_response(1), _make_response(2), _make_response(3)]
    mock_ai_client.return_value = client

    result = generate(
        question="prime",
        question_content="Write prime checker",
        provider="openai",
        model="gpt-4o-mini",
        iterations=3,
        data_dir=tmp_path / "data",
    )

    exp_id = result["id"]
    real_store = ExperimentStore(data_dir=tmp_path / "data")
    records = list(real_store.read_iterations(exp_id))
    assert len(records) == 3
    assert result["summary"]["successful_iterations"] == 3
    assert result["summary"]["failed_iterations"] == 0


@patch("faas_gauge.experiment.retry.AIClient")
def test_retry_failed_replaces_last_iteration_record(mock_ai_client: MagicMock, tmp_path) -> None:
    store = _make_store(tmp_path / "data")
    exp_id = "exp_retry"
    store.create_experiment(
        {
            "id": exp_id,
            "question_name": "prime",
            "question_content": "Write prime checker",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "iterations": 3,
        }
    )
    store.append_iteration(
        exp_id,
        {
            "iteration": 1,
            "python_code": "print(1)",
            "input_tokens": 1,
            "output_tokens": 5,
            "rtt_sec": 0.1,
            "tokens_per_second": 50.0,
            "error": None,
        },
    )
    store.append_iteration(
        exp_id,
        {
            "iteration": 2,
            "python_code": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "rtt_sec": 0.1,
            "tokens_per_second": None,
            "error": "rate limit",
        },
    )
    store.append_iteration(
        exp_id,
        {
            "iteration": 3,
            "python_code": "print(3)",
            "input_tokens": 1,
            "output_tokens": 5,
            "rtt_sec": 0.1,
            "tokens_per_second": 50.0,
            "error": None,
        },
    )

    client = MagicMock()
    client.ask.return_value = _make_response(2, error=None, output_tokens=8)
    mock_ai_client.return_value = client

    result = retry_failed(exp_id, data_dir=tmp_path / "data")

    records = list(store.read_iterations(exp_id))
    assert len(records) == 3
    assert records[1]["iteration"] == 2
    assert records[1]["error"] is None
    assert records[1].get("is_retry") is True
    assert result["retried"] == 1
    assert result["newly_successful"] == 1
    assert result["still_failed"] == 0
