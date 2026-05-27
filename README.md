# FaaS-GAUGE

FaaS-GAUGE is the repeatable benchmark for evaluating LLM-generated serverless code on AWS Lambda. It prompts multiple LLM providers to write Python AWS Lambda functions, validates the generated code (locally or on live Lambda), and stores structured results for analysis. FaaS-GAUGE is packaged here as a public artifact for the FaaS-GAUGE benchmark paper.

## Quickstart Test

Install the benchmark, copy and fill in your credentials file, generate code for the bundled prime-number question, validate it locally, then optionally run the full AWS Lambda pipeline. Results are exported as CSVs via `scripts/export_results.py`.

---

## File Layout

```
faas-gauge/
├── faas_gauge/           # Main Python package (ai, config, experiment, store, validator)
├── scripts/              # CLI wrappers and analysis scripts
├── tests/                # pytest suite
├── docs/                 # Procedure documents
├── samples/              # Sample CSV report outputs
├── data/
│   ├── config/
│   │   └── credentials.json.example   # Copy this to credentials.json and fill in keys
│   ├── prompts/
│   │   └── default.txt                # System prompt template sent to all models
│   ├── questions/
│   │   └── prime_number_generator.txt # Bundled example question
│   └── test_data/
│       └── prime_number_generator.json # Expected I/O for correctness checking
├── pyproject.toml
└── README.md
```

---

## Install

```bash
git clone https://github.com/KiritoMiao/FaaS-GAUGE.git faas-gauge && cd faas-gauge
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev,aws,notebooks]
```

Available required library bundles:

| Bundle      | What it adds                                    |
|-------------|-------------------------------------------------|
| `dev`       | pytest, pytest-cov, black, mypy                 |
| `aws`       | boto3 (required for AWS Lambda validation)      |
| `notebooks` | jupyter, pandas, matplotlib                     |
| `migrate`   | pyyaml (legacy credentials migration only)      |

---

## Configure Credentials

```bash
cp data/config/credentials.json.example data/config/credentials.json
# Edit data/config/credentials.json with your keys
```

The credentials file structure:

```json
{
  "ai_providers": {
    "openai": {
      "api_base": "https://api.openai.com/v1",
      "api_key": "sk-your-key-here",
      "system_prompt_id": "developer",
      "models": ["gpt-4o", "gpt-4o-mini"],
      "chain_as_system_models": ["o1", "o1-mini", "o3-mini"],
      "completion_models": [],
      "timeout": 30,
      "max_retries": 3
    }
  },
  "aws": {
    "profile": "default",
    "region": "us-west-2"
  },
  "settings": {
    "log_level": "WARNING",
    "default_timeout": 30,
    "system_prompt_file": "prompts/default.txt"
  }
}
```

Each key under `ai_providers` is a provider name (`openai`, `deepseek`, `xai`, `router/openrouter`, etc.). The `api_base` field lets you point at any provider or at an OpenAI-compatible endpoint.

### Environment Variable Overrides

| Variable | Description |
|---|---|
| `FAAS_GAUGE_DATA_DIR` | Override the auto-discovered `data/` directory path |
| `FAAS_GAUGE_CREDENTIALS` | Override the auto-discovered `credentials.json` path |
| `FAAS_GAUGE_AI_OPENAI_API_KEY` | Override the OpenAI API key from the credentials file |
| `FAAS_GAUGE_AI_{PROVIDER}_API_KEY` | Override any provider's API key (uppercase provider name) |

Data directory auto-discovery: the framework walks up from the CWD to look for `data/config/`. Set `FAAS_GAUGE_DATA_DIR` to skip discovery.

---

## The Prompts

`data/prompts/default.txt` is the system prompt sent to every model:

```
You are a helpful coding assistant. Write clean, efficient Python code.
```

`data/questions/prime_number_generator.txt` provides a bundled question as a working example:

```
Write an AWS Lambda function in Python that generates all prime numbers up to a given
number. A prime number is a natural number greater than 1 that has exactly two distinct
factors: 1 and itself (for example, 2 and 3 are prime, but 4 is not since it is
divisible by 2).

Requirements:
Input:  {"n": <upper_bound>}
Output: {"count": <total_primes>, "data": [<prime>, ...]}

Example: input {"n": 10} -> output {"count": 4, "data": [2, 3, 5, 7]}
```

The expected I/O pairs for correctness checking live in `data/test_data/prime_number_generator.json`.

---

## End-to-End Working Example - Prime Number Generation

The following commands run the full LLM code generation analysis pipeline on the bundled prime-number question.

### 1. Generate Code

```bash
.venv/bin/python scripts/generate.py \
    -f data/questions/prime_number_generator.txt \
    -p openai -m gpt-4o -n 3 -w 2 -g demo_run
```

Flag reference:

| Flag | Description |
|------|-------------|
| `-f` | Path to the question file |
| `-p` | Provider name (must match a key in `ai_providers`) |
| `-m` | Model name |
| `-n` | Number of iterations (code generations) per model |
| `-w` | Worker threads for parallel generation |
| `-g` | Test group tag (used to group related experiments) |

Output is written to `data/experiments/{experiment_id}/`:
- `experiment.json` — metadata (provider, model, question, timestamps, summary counts)
- `iterations.jsonl` — one JSON record per generation attempt (code, tokens, timing, errors)

### 2. Validate Locally

```bash
.venv/bin/python scripts/validate.py -g demo_run -r local
```

The validator injects SAAF instrumentation into each generated function, executes it in-process with `importlib`, and compares output against `data/test_data/prime_number_generator.json`. Each run is classified as one of the following states (states are ordered by increasing code quality):

```
syntax-error > out-of-memory > scaling-bug > timeout > functional-error > success
```

Results are written to `data/validations/{batch_id}/`:
- `batch.json` — batch metadata
- `runs.jsonl` — one record per (experiment, iteration) pair

### 3. Validate on AWS Lambda

**Prerequisites:**
- AWS account with an IAM role that has `AWSLambdaBasicExecutionRole`
- Region and profile set in `data/config/credentials.json` under the `aws` key

```bash
.venv/bin/python scripts/validate.py -g demo_run -r aws
```

The AWS runner:
1. Bundles the generated function with any declared dependencies via `pip install --target`
2. Deploys to Lambda using the configured IAM role
3. Invokes the function with test payloads
4. Parses CloudWatch `REPORT` lines for Max Memory Used
5. Cleans up the Lambda function after validation

Additional flags for AWS validation:

| Flag | Description |
|------|-------------|
| `-n` | Invocations per Lambda function |
| `-t` | Timeout in seconds |
| `--aws-memory` | Lambda memory allocation in MB |

### 4. Read Results

```bash
.venv/bin/python scripts/export_results.py
```

Exports CSVs to `data/reports/`. Pass `--test-group <tag>` to filter by group and `--report all` for all report types. Sample output formats are in `samples/`.

---

## Run the Full Weekly Pipeline (Optional)

```bash
./scripts/run_weekly_test.sh <week_tag>
```

Example: `./scripts/run_weekly_test.sh week1`

Flags: `--skip-generate`, `--skip-static`, `--skip-validate`, `--skip-reports`

**Warning:** `scripts/run_weekly_test.sh` and `scripts/rq4_analysis.py` are copied verbatim from the paper's full pipeline and reference questions (`car_position`, `distinct_integer_counter`, `minimal_cost_split`, etc.) that are **not bundled** in this public artifact. Running the script as-is will fail for those questions. To use these scripts:

- Either restrict the `QUESTIONS` array in `run_weekly_test.sh` to `prime_number_generator` only, or
- Add the missing question files yourself (see "Adding More Questions" below).

---

## Adding More Questions

New questions can be added to FaaS-GAUGE. Each new question requires:

1. `data/questions/{name}.txt` — the natural-language prompt sent to the LLM
2. `data/test_data/{name}.json` — expected I/O pairs for correctness checking (array of `{"input": {...}, "output": {...}}` objects)
3. (Optional) A custom validator in `faas_gauge/validator/validators.py` under the `VALIDATORS` dict, is required if generic output comparison is insufficient. This applies to functions that produce non-deterministic output. Validators must conform to the signature `(actual, expected, optional_arg=None) -> (bool, str)`.
4. Update the `QUESTIONS` array and `VALIDATION_CONFIG` in `scripts/run_weekly_test.sh`.
5. Update the pricing table and question list in `scripts/rq4_analysis.py`.

See `faas_gauge/validator/validators.py` and the `CLAUDE.md` file (present on disk, gitignored) for the full pattern.

---

## FaaS-Gauge Framework Test

```bash
pytest
```

This performs an offline test to verify if the FaaS-GAUGE framework is installed properly and ready to use. 
Tests mock `openai.OpenAI` and `boto3.Session` — no live API calls are made. All tests run from the repo root; configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

Run a single module or test:

```bash
pytest tests/test_experiment.py
pytest tests/test_experiment.py::test_generate
pytest -k classify
```

---

## Lint / Format / Types

FaaS-GAUGE uses three tools to test the FaaS-GAUGE framework implementation for errors, bugs, and sylisyic inconsistences.

```bash
ruff check           # lint
black .              # format (line-length 100, targets py310-py312)
mypy faas_gauge      # type-check
```

---

## Architecture Overview

```
config -> ai -> store -> validator -> experiment -> scripts/
```

| Module | Role |
|--------|------|
| `faas_gauge.config` | `get_data_dir()`, `load_credentials()`. Honors `FAAS_GAUGE_DATA_DIR` / `FAAS_GAUGE_CREDENTIALS`. |
| `faas_gauge.ai` | `AIClient.ask()` against any OpenAI-compatible endpoint. Errors are returned as `AIResponse(error=...)`, never raised. |
| `faas_gauge.store` | `ExperimentStore` — JSON metadata + JSONL append-only iterations/runs. |
| `faas_gauge.validator` | `FunctionValidator` with `LocalRunner` (in-process) and `AWSRunner` (Lambda deploy + invoke). SAAF instrumentation injected at build time. |
| `faas_gauge.experiment` | `generate()` and `validate_batch()` with resume support. `classify_function_status` priority order is load-bearing. |
| `scripts/` | Thin argparse wrappers over library functions + analysis/reporting scripts. |
