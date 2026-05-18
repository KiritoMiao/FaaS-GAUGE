#!/usr/bin/env bash
# ==========================================================================
#  Weekly Test Script — generate, static-analyse, validate (AWS), report
#
#  Usage:
#    ./scripts/run_weekly_test.sh <week_tag>
#    ./scripts/run_weekly_test.sh week4
#
#  Requirements:
#    - .venv with pylint, radon, boto3 installed
#    - AWS credentials configured (for Lambda validation)
#    - OpenRouter / OpenAI / xAI API keys in data/config/credentials.json
# ==========================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
QDIR="$PROJECT_ROOT/data/questions"
LOGDIR="$PROJECT_ROOT/data/logs"

# Models (provider|model)
MODELS=(
  "openai|gpt-5.2"
  "openai|gpt-5-mini"
  "router/openrouter|anthropic/claude-opus-4.5"
  "router/openrouter|anthropic/claude-sonnet-4.5"
  "router/openrouter|google/gemini-3-flash-preview"
  "xai|grok-4-1-fast-reasoning"
)

# Questions to generate and test
QUESTIONS=(
  "prime_number_generator"
  "distinct_integer_counter"
  "car_position"
  "minimal_cost_split"
)

# Generation iterations
GEN_ITERATIONS=10

# AWS validation config per question
# Format: question|iterations|timeout_sec|aws_memory_mb
VALIDATION_CONFIG=(
  "prime_number_generator|1|300|1769"
  "car_position|2|120|1769"
  "minimal_cost_split|2|120|1769"
  # distinct_integer_counter has no test data — skip validation
)

# ── Argument parsing ──────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
  echo "Usage: $0 <week_tag> [--skip-generate] [--skip-static] [--skip-validate] [--skip-reports]"
  echo ""
  echo "Example:  $0 week4"
  echo "          $0 week4 --skip-generate  # re-run validation & reports only"
  exit 1
fi

WEEK_TAG="$1"
shift

SKIP_GENERATE=false
SKIP_STATIC=false
SKIP_VALIDATE=false
SKIP_REPORTS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-generate)  SKIP_GENERATE=true ;;
    --skip-static)    SKIP_STATIC=true ;;
    --skip-validate)  SKIP_VALIDATE=true ;;
    --skip-reports)   SKIP_REPORTS=true ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOGDIR/${WEEK_TAG}_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

# ── Phase 1: Generation ──────────────────────────────────────────────────
if [ "$SKIP_GENERATE" = false ]; then
  log "════════════════════════════════════════════════════════════"
  log "Phase 1: Generating code ($WEEK_TAG)"
  log "  Questions: ${QUESTIONS[*]}"
  log "  Models: ${#MODELS[@]}"
  log "  Iterations: $GEN_ITERATIONS"
  log "════════════════════════════════════════════════════════════"

  GEN_LOGDIR="$LOGDIR/${WEEK_TAG}_gen_${TIMESTAMP}"
  mkdir -p "$GEN_LOGDIR"

  PIDS=()
  for q in "${QUESTIONS[@]}"; do
    for pm in "${MODELS[@]}"; do
      IFS='|' read -r provider model <<< "$pm"
      logfile="$GEN_LOGDIR/${q}__${model//\//-}.log"
      $PYTHON "$PROJECT_ROOT/scripts/generate.py" \
        -f "$QDIR/${q}.txt" \
        -p "$provider" \
        -m "$model" \
        -n "$GEN_ITERATIONS" \
        -g "$WEEK_TAG" \
        > "$logfile" 2>&1 &
      PIDS+=($!)
    done
  done

  log "Launched ${#PIDS[@]} generation jobs in parallel..."
  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAIL++))
  done

  # Summarise
  TOTAL_SUCCESS=0
  TOTAL_FAILED=0
  for logfile in "$GEN_LOGDIR"/*.log; do
    if grep -q "Summary:" "$logfile" 2>/dev/null; then
      s=$(grep "Summary:" "$logfile" | sed 's/.*success=\([0-9]*\).*/\1/')
      f=$(grep "Summary:" "$logfile" | sed 's/.*failed=\([0-9]*\).*/\1/')
      TOTAL_SUCCESS=$((TOTAL_SUCCESS + s))
      TOTAL_FAILED=$((TOTAL_FAILED + f))
    fi
  done
  log "Generation complete: ${TOTAL_SUCCESS} success, ${TOTAL_FAILED} failed, ${FAIL} job errors"
  log "Logs: $GEN_LOGDIR/"
else
  log "Skipping generation (--skip-generate)"
fi

# ── Phase 2: Static Analysis ─────────────────────────────────────────────
if [ "$SKIP_STATIC" = false ]; then
  log ""
  log "════════════════════════════════════════════════════════════"
  log "Phase 2: Static Analysis ($WEEK_TAG)"
  log "════════════════════════════════════════════════════════════"

  $PYTHON "$PROJECT_ROOT/scripts/static_analysis.py" -g "$WEEK_TAG" -w 8 2>&1 | tee -a "$MAIN_LOG"
  log "Static analysis complete."
else
  log "Skipping static analysis (--skip-static)"
fi

# ── Phase 3: AWS Validation ───────────────────────────────────────────────
if [ "$SKIP_VALIDATE" = false ]; then
  log ""
  log "════════════════════════════════════════════════════════════"
  log "Phase 3: AWS Validation ($WEEK_TAG)"
  log "════════════════════════════════════════════════════════════"

  for vc in "${VALIDATION_CONFIG[@]}"; do
    IFS='|' read -r vq viters vtimeout vmem <<< "$vc"
    log "Validating: $vq (iters=$viters, timeout=${vtimeout}s, memory=${vmem}MB)"
    $PYTHON "$PROJECT_ROOT/scripts/validate.py" \
      -g "$WEEK_TAG" \
      -q "$vq" \
      -r aws \
      -n "$viters" \
      -t "$vtimeout" \
      --aws-memory "$vmem" \
      2>&1 | tee -a "$MAIN_LOG"
    log "$vq validation complete."
  done
else
  log "Skipping validation (--skip-validate)"
fi

# ── Phase 4: Export Reports ───────────────────────────────────────────────
if [ "$SKIP_REPORTS" = false ]; then
  log ""
  log "════════════════════════════════════════════════════════════"
  log "Phase 4: Exporting Reports ($WEEK_TAG)"
  log "════════════════════════════════════════════════════════════"

  $PYTHON "$PROJECT_ROOT/scripts/export_results.py" \
    --report all \
    --test-group "$WEEK_TAG" \
    2>&1 | tee -a "$MAIN_LOG"
  log "Reports exported."
else
  log "Skipping reports (--skip-reports)"
fi

# ── Summary ───────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════"
log "Weekly test complete: $WEEK_TAG"
log "Log: $MAIN_LOG"
log "════════════════════════════════════════════════════════════"
