#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR/Week3"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_DIR/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-5e-4}"
LR_DECAY_FACTOR="${LR_DECAY_FACTOR:-0.5}"
LR_DECAY_PATIENCE="${LR_DECAY_PATIENCE:-2}"
MIN_LR="${MIN_LR:-1e-6}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
MAX_LEN_CHAR="${MAX_LEN_CHAR:-100}"
MAX_LEN_WORD="${MAX_LEN_WORD:-40}"
VOCAB_SIZE="${VOCAB_SIZE:-5000}"
MIN_FREQ="${MIN_FREQ:-2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"
EVAL_SPLIT="${EVAL_SPLIT:-heldout}"
TRAINABLE_BACKBONE="${TRAINABLE_BACKBONE:-0}"
SCHEDULED_SAMPLING="${SCHEDULED_SAMPLING:-0}"
SCHEDULED_SAMPLING_MAX_RATIO="${SCHEDULED_SAMPLING_MAX_RATIO:-0.25}"

mkdir -p "$RUNS_DIR"

if [[ ! -f "$DATA_DIR/annotations/train.json" ]]; then
  echo "Missing training annotations: $DATA_DIR/annotations/train.json" >&2
  exit 1
fi

echo "Training configuration"
echo "  DATA_DIR=$DATA_DIR"
echo "  RUNS_DIR=$RUNS_DIR"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  LR=$LR"
echo "  LR_DECAY_FACTOR=$LR_DECAY_FACTOR"
echo "  LR_DECAY_PATIENCE=$LR_DECAY_PATIENCE"
echo "  MIN_LR=$MIN_LR"
echo "  EARLY_STOPPING_PATIENCE=$EARLY_STOPPING_PATIENCE"
echo "  MAX_LEN_CHAR=$MAX_LEN_CHAR"
echo "  MAX_LEN_WORD=$MAX_LEN_WORD"
echo "  EVAL_SPLIT=$EVAL_SPLIT"
echo "  PRETRAINED_ENCODER=1 (forced)"
echo "  TRAINABLE_BACKBONE=$TRAINABLE_BACKBONE"
echo "  SCHEDULED_SAMPLING=$SCHEDULED_SAMPLING"
echo "  SCHEDULED_SAMPLING_MAX_RATIO=$SCHEDULED_SAMPLING_MAX_RATIO"
echo "  VAL_RATIO=$VAL_RATIO"
echo "  SPLIT_SEED=$SPLIT_SEED"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --lr-decay-factor "$LR_DECAY_FACTOR"
  --lr-decay-patience "$LR_DECAY_PATIENCE"
  --min-lr "$MIN_LR"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --vocab-size "$VOCAB_SIZE"
  --min-freq "$MIN_FREQ"
  --embedding-dim "$EMBEDDING_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --num-workers "$NUM_WORKERS"
  --val-ratio "$VAL_RATIO"
  --split-seed "$SPLIT_SEED"
  --pretrained-encoder
)

if [[ -n "$LIMIT_TRAIN" ]]; then
  COMMON_ARGS+=(--limit-train "$LIMIT_TRAIN")
fi

if [[ -n "$LIMIT_VAL" ]]; then
  COMMON_ARGS+=(--limit-val "$LIMIT_VAL")
fi

if [[ "$TRAINABLE_BACKBONE" == "1" ]]; then
  COMMON_ARGS+=(--trainable-backbone)
fi

if [[ "$SCHEDULED_SAMPLING" == "1" ]]; then
  COMMON_ARGS+=(--scheduled-sampling --scheduled-sampling-max-ratio "$SCHEDULED_SAMPLING_MAX_RATIO")
fi

EVAL_ARGS=(
  --data-dir "$DATA_DIR"
  --split "$EVAL_SPLIT"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$MAX_EVAL_EXAMPLES" ]]; then
  EVAL_ARGS+=(--max-examples "$MAX_EVAL_EXAMPLES")
fi

run_experiment() {
  local name="$1"
  local token_level="$2"
  local max_len="$3"
  shift 3

  local output_dir="$RUNS_DIR/$name"
  echo
  echo "============================================================"
  echo "Training experiment: $name"
  echo "Output directory: $output_dir"
  echo "Token level: $token_level"
  echo "Max length: $max_len"
  echo "============================================================"

  "$PYTHON_BIN" "$PROJECT_DIR/main.py" train \
    --output-dir "$output_dir" \
    --token-level "$token_level" \
    --max-len "$max_len" \
    "${COMMON_ARGS[@]}" \
    "$@"

  echo "Evaluating best checkpoint on internal held-out split for: $name"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" eval \
    --checkpoint "$output_dir/best.pt" \
    --output-json "$output_dir/eval_${EVAL_SPLIT}.json" \
    "${EVAL_ARGS[@]}"
}

run_experiment "baseline_resnet18_gru_char" "char" "$MAX_LEN_CHAR" \
  --encoder resnet18 \
  --decoder gru

run_experiment "encoder_resnet34_gru_char" "char" "$MAX_LEN_CHAR" \
  --encoder resnet34 \
  --decoder gru

run_experiment "decoder_resnet18_lstm_char" "char" "$MAX_LEN_CHAR" \
  --encoder resnet18 \
  --decoder lstm

run_experiment "token_resnet18_gru_word" "word" "$MAX_LEN_WORD" \
  --encoder resnet18 \
  --decoder gru

run_experiment "attention_resnet18_gru_char" "char" "$MAX_LEN_CHAR" \
  --encoder resnet18 \
  --decoder gru \
  --use-attention

run_experiment "combined_resnet34_lstm_word_attention" "word" "$MAX_LEN_WORD" \
  --encoder resnet34 \
  --decoder lstm \
  --use-attention

SUMMARY_PATH="$RUNS_DIR/experiment_summary.tsv"
export RUNS_DIR SUMMARY_PATH

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

runs_dir = Path(os.environ["RUNS_DIR"])
summary_path = Path(os.environ["SUMMARY_PATH"])
rows = []

for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
    history_path = run_dir / "history.json"
    history_payload = json.loads(history_path.read_text()) if history_path.exists() else {}
    eval_candidates = sorted(run_dir.glob("eval_*.json"))
    eval_payload = json.loads(eval_candidates[0].read_text()) if eval_candidates else {}
    metrics = eval_payload.get("metrics", {})
    config = eval_payload.get("config", {}) or history_payload.get("config", {})
    history = history_payload.get("history", [])
    best_epoch = None
    if history:
        best_epoch = max(
            history,
            key=lambda row: row.get("meteor", 0.0) + row.get("rougeL", 0.0) + row.get("bleu2", 0.0),
        ).get("epoch")
    rows.append(
        [
            run_dir.name,
            config.get("encoder_name", ""),
            config.get("decoder_type", ""),
            config.get("token_level", ""),
            str(config.get("use_attention", False)),
            str(config.get("pretrained_encoder", False)),
            str(config.get("scheduled_sampling", False)),
            str(config.get("scheduled_sampling_max_ratio", "")),
            str(config.get("lr", "")),
            str(config.get("lr_decay_factor", "")),
            str(config.get("lr_decay_patience", "")),
            str(config.get("early_stopping_patience", "")),
            str(best_epoch if best_epoch is not None else ""),
            f"{metrics.get('bleu1', 0.0):.4f}",
            f"{metrics.get('bleu2', 0.0):.4f}",
            f"{metrics.get('rougeL', 0.0):.4f}",
            f"{metrics.get('meteor', 0.0):.4f}",
        ]
    )

header = [
    "run",
    "encoder",
    "decoder",
    "token_level",
    "attention",
    "pretrained_encoder",
    "scheduled_sampling",
    "scheduled_sampling_max_ratio",
    "lr",
    "lr_decay_factor",
    "lr_decay_patience",
    "early_stopping_patience",
    "best_epoch",
    "bleu1",
    "bleu2",
    "rougeL",
    "meteor",
]

with summary_path.open("w", encoding="utf-8") as handle:
    handle.write("\t".join(header) + "\n")
    for row in rows:
        handle.write("\t".join(row) + "\n")

print(f"Saved summary to {summary_path}")
PY

echo
echo "All experiments finished."
echo "Summary table: $SUMMARY_PATH"
