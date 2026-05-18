# Weekly Test Procedure

End-to-end procedure for running a weekly AI code generation test cycle.

## Prerequisites

- Python 3.13+ via `.venv/bin/python`
- AWS credentials configured (`data/config/credentials.json`)
- OpenRouter / OpenAI / xAI API credits available
- Sufficient disk space (validation JSONL files can reach 1–4 GB per batch)

## Models

| Provider | Model | CLI provider value |
|---|---|---|
| OpenAI | gpt-5.2 | `openai` |
| OpenAI | gpt-5-mini | `openai` |
| OpenRouter | anthropic/claude-opus-4.5 | `router/openrouter` |
| OpenRouter | anthropic/claude-sonnet-4.5 | `router/openrouter` |
| OpenRouter | google/gemini-3-flash-preview | `router/openrouter` |
| xAI | grok-4-1-fast-reasoning | `xai` |

## Questions

| Question | Validation type | AWS test profile |
|---|---|---|
| `prime_number_generator` | Size-based performance (9 sizes: 1M–200M) | 1 iteration per size, 300 s timeout, 1769 MB |
| `car_position` | Target-30 s repeated (136 iterations × 2 runs) | 136 iterations, 120 s per-invocation timeout, 300 s total budget |
| `minimal_cost_split` | Target-30 s repeated (666 iterations × 2 runs) | 666 iterations, 120 s per-invocation timeout, 300 s total budget |
| `distinct_integer_counter` | No test data — **cannot be validated on AWS** |

---

## Step 1 — Generate code (all questions × all models × 10 iterations)

```bash
WEEK=week4   # change per week

MODELS=(
  "openai gpt-5.2"
  "openai gpt-5-mini"
  "router/openrouter anthropic/claude-opus-4.5"
  "router/openrouter anthropic/claude-sonnet-4.5"
  "router/openrouter google/gemini-3-flash-preview"
  "xai grok-4-1-fast-reasoning"
)
QUESTIONS=(prime_number_generator distinct_integer_counter car_position minimal_cost_split)

for q in "${QUESTIONS[@]}"; do
  for m in "${MODELS[@]}"; do
    set -- $m
    provider=$1; model=$2
    nohup .venv/bin/python scripts/generate.py \
      -q "$q" -p "$provider" -m "$model" \
      -g "$WEEK" -n 10 \
      > "data/logs/${WEEK}_gen_${q}_${model//\//-}.log" 2>&1 &
  done
done
wait
```

Monitor: `wc -l data/logs/${WEEK}_gen_*.log`

Expected output: 24 experiments (4 questions × 6 models), 240 iterations total.

---

## Step 2 — Static analysis

```bash
.venv/bin/python scripts/static_analysis.py -g "$WEEK"
```

Runs pylint + radon on every iteration of every experiment in the group.

---

## Step 3 — AWS validation

### 3a. Prime number generator

Single run, 1 iteration per size test, 300 s per-invocation timeout:

```bash
.venv/bin/python -u scripts/validate.py \
  -g "$WEEK" -r aws -n 1 -t 300 \
  -q prime_number_generator --aws-memory 1769 \
  | tee "data/logs/${WEEK}_validate_prime.log"
```

Expected: ~59–60 iterations validated, 9 size tests each. Large sizes (150M+) may OOM.

### 3b. Car position — run 1

```bash
nohup .venv/bin/python -u scripts/validate.py \
  -g "$WEEK" -r aws -n 136 -t 120 \
  -q car_position --aws-memory 1769 \
  --max-total-seconds 300 \
  > "data/logs/${WEEK}_validate_car_position_run1.log" 2>&1 &
```

### 3c. Minimal cost split — run 1

```bash
nohup .venv/bin/python -u scripts/validate.py \
  -g "$WEEK" -r aws -n 666 -t 120 \
  -q minimal_cost_split --aws-memory 1769 \
  --max-total-seconds 300 \
  > "data/logs/${WEEK}_validate_minimal_cost_split_run1.log" 2>&1 &
```

### 3d. Wait for run 1 to finish

Monitor progress:

```bash
# Count completed iterations per batch
for d in $(ls -dt data/validations/val_*); do
  batch=$(basename "$d")
  n=$(wc -l < "$d/runs.jsonl" 2>/dev/null || echo 0)
  echo "$batch: $n runs"
done
```

Or tail the logs:

```bash
tail -f data/logs/${WEEK}_validate_car_position_run1.log
```

### 3e. Car position — run 2

After run 1 completes, start run 2 with `--no-resume`:

```bash
nohup .venv/bin/python -u scripts/validate.py \
  -g "$WEEK" -r aws -n 136 -t 120 \
  -q car_position --aws-memory 1769 \
  --max-total-seconds 300 --no-resume \
  > "data/logs/${WEEK}_validate_car_position_run2.log" 2>&1 &
```

### 3f. Minimal cost split — run 2

After run 1 completes:

```bash
nohup .venv/bin/python -u scripts/validate.py \
  -g "$WEEK" -r aws -n 666 -t 120 \
  -q minimal_cost_split --aws-memory 1769 \
  --max-total-seconds 300 --no-resume \
  > "data/logs/${WEEK}_validate_minimal_cost_split_run2.log" 2>&1 &
```

### Timing estimates

| Question | Run 1 | Run 2 | Total |
|---|---|---|---|
| prime_number_generator | 1–3 h | n/a | 1–3 h |
| car_position | 30–45 min | 30–45 min | 1–1.5 h |
| minimal_cost_split | 2–3 h | 2–3 h | 4–6 h |

---

## Step 4 — Generate reports

### 4a. Standard reports (static analysis, performance, merged, composite)

```bash
.venv/bin/python scripts/export_results.py -g "$WEEK" --report-type static_analysis
.venv/bin/python scripts/export_results.py -g "$WEEK" --report-type performance
.venv/bin/python scripts/export_results.py -g "$WEEK" --report-type merged
.venv/bin/python scripts/export_results.py -g "$WEEK" --report-type composite
```

### 4b. Per-question weekly result CSVs

```bash
.venv/bin/python scripts/generate_weekly_results.py -g "$WEEK"
```

Output files (`data/reports/weekly/`):

| File | Format |
|---|---|
| `{WEEK}_prime_number_generator_lambda_performance.csv` | Pivot by size (1m–200m columns for time + RAM) |
| `{WEEK}_car_position_aws_target30_avg2_detailed.csv` | Target-30 s detailed (run 1/run 2 times, averaged) |
| `{WEEK}_minimal_cost_split_aws_target30_avg2_detailed.csv` | Target-30 s detailed (run 1/run 2 times, averaged) |

---

## Step 5 — Verify results

```bash
# Check row counts (should be iterations + header + 8 stats rows)
wc -l data/reports/weekly/${WEEK}_*.csv

# Quick sanity check
head -3 data/reports/weekly/${WEEK}_car_position_aws_target30_avg2_detailed.csv
```

### Expected CSV columns

**Prime (size-pivot):**
`model_provider, seq, maximum_successful_prime, function_status_200m, 1m, 10m, 50m, 75m, 100m, 125m, 150m, 175m, 200m, 1m_max_ram, ...`

**Target-30 (car_position / minimal_cost_split):**
`question, model_provider, seq, function_status, performance_status, run_1_status, run_1_total_ms, run_1_max_ram_mb, run_2_status, run_2_total_ms, run_2_max_ram_mb, avg_of_runs_total_ms, avg_of_runs_total_seconds, max_ram_mb, input_url, validation_message, first_execution_error`

---

## CLI reference

### `scripts/validate.py`

| Flag | Default | Description |
|---|---|---|
| `-g` / `--test-group` | required | Test group name (e.g., `week2`) |
| `-r` / `--runtime` | `local` | `local` or `aws` |
| `-n` / `--iterations` | 1 | Iterations per performance test |
| `-t` / `--max-runtime` | 30 | Per-invocation timeout (seconds) |
| `-q` / `--question` | None | Filter to a single question |
| `--aws-memory` | 1769 | Lambda memory (MB) |
| `--max-total-seconds` | None | Total time budget for performance iterations per code version |
| `--no-resume` | false | Skip resume logic; re-validate all iterations |
| `--no-reuse` | false | Don't reuse deployed Lambda functions |

### `scripts/generate_weekly_results.py`

| Flag | Default | Description |
|---|---|---|
| `-g` / `--group` | required | Test group to report on |
| `-o` / `--output-dir` | `data/reports/weekly` | Output directory |
| `-d` / `--data-dir` | `data` | Data directory root |

---

## How target-30 s works

For `car_position` and `minimal_cost_split`, the goal is to repeatedly invoke the function enough times that the **total execution time** across all invocations approaches ~30 seconds. This gives a stable aggregate measurement.

1. The validator calls Lambda N times sequentially (N = `--iterations`).
2. Each invocation has a per-call timeout (`--max-runtime`, default 120 s).
3. An optional `--max-total-seconds` budget caps wall-clock time across all N invocations per code version. If the budget is exhausted, the loop stops early and only completed invocations are recorded.
4. Two independent runs are performed (run 1 then run 2 with `--no-resume`) to produce two total-time measurements.
5. The reporting script (`generate_weekly_results.py`) pairs the two runs, sums execution times from the `performance` test in each, and averages them.

### Iteration counts (calibrated from week 1)

| Question | Iterations | Rationale |
|---|---|---|
| car_position | 136 | ~260 ms per call × 136 ≈ 35 s |
| minimal_cost_split | 666 | ~45 ms per call × 666 ≈ 30 s (fast models); slow models capped at 300 s budget |

### Resume vs no-resume

- **Run 1** uses resume (default): if the process crashes mid-way, restarting it will skip already-completed iterations.
- **Run 2** uses `--no-resume`: forces a fresh pass over all iterations to produce a second independent measurement.
- Resume matches on `(test_group, runtime, question_filter, iterations_per_test)`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `runs.jsonl` stuck at 0 | Lambda deploy / build failure | Check log file; may need `--no-reuse` |
| Two batches same timestamp | Parallel starts within same second | Stagger with `sleep 2` |
| Validation skips everything | Resume found prior batch | Use `--no-resume` for run 2 |
| OOM on large prime sizes | Lambda 1769 MB insufficient for 175M+ | Expected; recorded as `out-of-memory` status |
| 0% pass rate for minimal_cost_split | Slow models produce O(n²) code | Expected; `performance_status=failed` means avg > 30 s |
| Huge JSONL files (1–4 GB) | High iteration count + output data | Normal; `generate_weekly_results.py` uses slim reader |
