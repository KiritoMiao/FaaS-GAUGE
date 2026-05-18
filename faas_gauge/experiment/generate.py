"""Code generation orchestration."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from faas_gauge.ai import AIClient
from faas_gauge.store import ExperimentStore


def _generate_experiment_id(
    question: str, provider: str, model: str, data_dir: Path
) -> str:
    """Generate a unique experiment ID with dedup suffix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Sanitize provider/model: replace '/' with '-' so the ID is path-safe
    safe_provider = provider.replace("/", "-")
    safe_model = model.replace("/", "-")
    base_id = f"{timestamp}+{question}+{safe_provider}+{safe_model}"

    experiments_dir = data_dir / "experiments"
    if not (experiments_dir / base_id).exists():
        return base_id

    suffix = 2
    while (experiments_dir / f"{base_id}__{suffix}").exists():
        suffix += 1
    return f"{base_id}__{suffix}"


def _run_single_iteration(
    client: AIClient,
    question_content: str,
    provider: str,
    model: str,
    iteration: int,
) -> dict[str, Any]:
    """Run a single AI iteration and return a result dict."""
    response = client.ask(question_content, provider, model)

    return {
        "iteration": iteration,
        "raw_result": response.raw_result,
        "python_code": response.python_code,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "tokens_per_second": response.tokens_per_second,
        "cost": response.cost,
        "rtt_sec": response.rtt_sec,
        "error": response.error,
        "timestamp": datetime.now().isoformat(),
    }


def generate(
    question: str,
    question_content: str,
    provider: str,
    model: str,
    iterations: int = 100,
    max_workers: int | None = None,
    test_group: str | None = None,
    data_dir: str | Path = "data",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Generate code by asking an AI model a question multiple times."""
    data_dir = Path(data_dir)
    store = ExperimentStore(data_dir=data_dir)

    experiment_id = _generate_experiment_id(question, provider, model, data_dir)

    meta: dict[str, Any] = {
        "id": experiment_id,
        "question_name": question,
        "question_content": question_content,
        "provider": provider,
        "model": model,
        "iterations": iterations,
    }
    if test_group:
        meta["test_group"] = test_group

    store.create_experiment(meta)

    client = AIClient(config_path=config_path)

    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0
    total_tps = 0.0
    successful = 0
    failed = 0

    if max_workers is None or max_workers <= 1:
        for i in range(1, iterations + 1):
            result = _run_single_iteration(client, question_content, provider, model, i)
            store.append_iteration(experiment_id, result)

            if result["error"]:
                failed += 1
            else:
                successful += 1
                total_input_tokens += result["input_tokens"]
                total_output_tokens += result["output_tokens"]
                total_time += result["rtt_sec"]
                if result["tokens_per_second"]:
                    total_tps += result["tokens_per_second"]
    else:

        def _create_and_run(iteration: int) -> dict[str, Any]:
            thread_client = AIClient(config_path=config_path)
            return _run_single_iteration(
                thread_client, question_content, provider, model, iteration
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_create_and_run, i): i for i in range(1, iterations + 1)
            }
            for future in as_completed(futures):
                result = future.result()
                store.append_iteration(experiment_id, result)

                if result["error"]:
                    failed += 1
                else:
                    successful += 1
                    total_input_tokens += result["input_tokens"]
                    total_output_tokens += result["output_tokens"]
                    total_time += result["rtt_sec"]
                    if result["tokens_per_second"]:
                        total_tps += result["tokens_per_second"]

    summary = {
        "total_iterations": iterations,
        "successful_iterations": successful,
        "failed_iterations": failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_time": round(total_time, 2),
        "total_tokens_per_second": round(total_tps, 2),
        "average_time": round(total_time / successful, 2) if successful > 0 else 0,
    }

    store.update_experiment_summary(experiment_id, summary)

    return {
        "id": experiment_id,
        "summary": summary,
    }
