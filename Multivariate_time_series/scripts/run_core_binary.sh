#!/usr/bin/env bash
set -e

# Minimal runner for binary tasks (e.g., ICU mortality, readmission, pancreatic cancer)
# Models: mlp | transformer | lstm | retain

DEFAULT_CONFIG="configs/mimiciv_icu_mortality.yaml"
CONFIG_FILE="$DEFAULT_CONFIG"
DATASET="default"
TASK="default"

# Parse flags (keep positional args compatible)
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)  CONFIG_FILE="$2"; shift 2;;
    -d|--dataset) DATASET="$2"; shift 2;;
    -t|--task)    TASK="$2"; shift 2;;
    *)            ARGS+=("$1"); shift;;
  esac
done
set -- "${ARGS[@]}"

COMMAND=${1:-train}      # train | eval
MODEL=${2:-transformer}
CHECKPOINT=${3:-}

if [[ "$COMMAND" != "train" && "$COMMAND" != "eval" ]]; then
  echo "Usage: $0 [train|eval] [mlp|transformer|lstm|retain] [checkpoint_if_eval] [-c CONFIG] [-d DATASET] [-t TASK]" >&2
  exit 1
fi

EXP_ROOT="experiments/${DATASET}/${TASK}"
EXP_LOG_DIR="${EXP_ROOT}/logs"
EXP_CKPT_DIR="${EXP_ROOT}/checkpoints"
DERIVED_CONFIG="${EXP_ROOT}/config.yaml"

mkdir -p "$EXP_LOG_DIR" "$EXP_CKPT_DIR"

# Derive a per-task config that rewrites logging/checkpoint dirs.
python3 - "$CONFIG_FILE" "$DERIVED_CONFIG" "$EXP_LOG_DIR" "$EXP_CKPT_DIR" <<'PYSNIP'
import sys, yaml
base, out, logd, ckptd = sys.argv[1:5]
with open(base, 'r') as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault('logging', {})['dir'] = logd
cfg.setdefault('checkpoint', {})['dir'] = ckptd
with open(out, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out)
PYSNIP

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [[ "$COMMAND" == "train" ]]; then
  LOG_FILE="${EXP_LOG_DIR}/${MODEL}_${TIMESTAMP}.out"
  echo "[INFO] Training: model=${MODEL} dataset=${DATASET} task=${TASK}"
  echo "[INFO] Base config:    ${CONFIG_FILE}"
  echo "[INFO] Derived config: ${DERIVED_CONFIG}"
  echo "[INFO] Shell log:      ${LOG_FILE}"

  nohup python3 src/train_core_models.py \
    --config "$DERIVED_CONFIG" \
    --model "$MODEL" \
    > "$LOG_FILE" 2>&1 &

  PID=$!
  echo "[INFO] Started (PID ${PID}). Tailing logs (Ctrl+C to stop tail; training continues)."
  tail -f "$LOG_FILE"

else
  if [[ -z "$CHECKPOINT" ]]; then
    echo "[ERROR] eval requires a checkpoint path as the 3rd positional argument." >&2
    exit 1
  fi
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi

  LOG_FILE="${EXP_LOG_DIR}/eval_${MODEL}_${TIMESTAMP}.out"
  echo "[INFO] Evaluating: model=${MODEL} dataset=${DATASET} task=${TASK}"
  echo "[INFO] Checkpoint: ${CHECKPOINT}"
  echo "[INFO] Derived config: ${DERIVED_CONFIG}"

  python3 src/train_core_models.py \
    --config "$DERIVED_CONFIG" \
    --model "$MODEL" \
    --eval-only \
    --checkpoint "$CHECKPOINT" \
    2>&1 | tee "$LOG_FILE"

  echo "[INFO] Eval log: ${LOG_FILE}"
fi
