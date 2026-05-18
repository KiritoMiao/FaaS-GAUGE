"""Retry failed iterations in an experiment."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from faas_gauge.ai import AIClient
from faas_gauge.store import ExperimentStore
from faas_gauge.store import layout as _layout


def retry_failed(
    experiment_id: str,
    max_workers: int | None = None,
    data_dir: str | Path = "data",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Retry failed iterations in an experiment."""
    del max_workers

    data_dir = Path(data_dir)
    store = ExperimentStore(data_dir=data_dir)

    meta = store.load_experiment(experiment_id)
    provider = meta["provider"]
    model = meta["model"]
    question_content = meta["question_content"]

    all_iterations = list(store.read_iterations(experiment_id))
    failed_nums = [
        it["iteration"]
        for it in all_iterations
        if it.get("error") is not None and it.get("output_tokens", 0) == 0
    ]

    if not failed_nums:
        return {
            "experiment_id": experiment_id,
            "retried": 0,
            "newly_successful": 0,
            "still_failed": 0,
        }

    client = AIClient(config_path=config_path)

    newly_successful = 0
    still_failed = 0

    for iteration_num in failed_nums:
        response = client.ask(question_content, provider, model)

        record = {
            "iteration": iteration_num,
            "raw_result": response.raw_result,
            "python_code": response.python_code,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "tokens_per_second": response.tokens_per_second,
            "cost": response.cost,
            "rtt_sec": response.rtt_sec,
            "error": response.error,
            "timestamp": datetime.now().isoformat(),
            "is_retry": True,
        }

        if response.error:
            still_failed += 1
        else:
            newly_successful += 1

        store.append_iteration(experiment_id, record)

    all_records = list(store.read_iterations(experiment_id))

    by_iteration: dict[int, dict[str, Any]] = {}
    for rec in all_records:
        by_iteration[rec["iteration"]] = rec

    iterations_file = _layout.iterations_path(data_dir, experiment_id)
    with open(iterations_file, "w", encoding="utf-8") as handle:
        for it_num in sorted(by_iteration.keys()):
            line = json.dumps(by_iteration[it_num], ensure_ascii=False, default=str)
            handle.write(line + "\n")

    final_records = list(store.read_iterations(experiment_id))
    successful = sum(1 for r in final_records if not r.get("error"))
    failed_count = sum(1 for r in final_records if r.get("error"))
    total_input = sum(r.get("input_tokens", 0) for r in final_records if not r.get("error"))
    total_output = sum(r.get("output_tokens", 0) for r in final_records if not r.get("error"))
    total_time = sum(r.get("rtt_sec", 0) for r in final_records if not r.get("error"))
    total_tps = sum(r.get("tokens_per_second", 0) or 0 for r in final_records if not r.get("error"))

    summary = {
        "total_iterations": len(final_records),
        "successful_iterations": successful,
        "failed_iterations": failed_count,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_time": round(total_time, 2),
        "total_tokens_per_second": round(total_tps, 2),
        "average_time": round(total_time / successful, 2) if successful > 0 else 0,
    }
    store.update_experiment_summary(experiment_id, summary)

    return {
        "experiment_id": experiment_id,
        "retried": len(failed_nums),
        "newly_successful": newly_successful,
        "still_failed": still_failed,
        "summary": summary,
    }
