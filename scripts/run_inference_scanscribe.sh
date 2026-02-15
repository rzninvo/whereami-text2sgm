#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

# Set these to match your local data layout
DATA_DIR="${WHEREAMI_DATA_ROOT:-./data}/processed_data"
CKPT_DIR="${WHEREAMI_DATA_ROOT:-./data}/model_checkpoints/graph2graph"

python -m whereami.models.inference \
  --graphs  "$DATA_DIR" \
  --ckpt    "$CKPT_DIR/model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint.pt" \
  --top_k   5 \
  --device  cuda \
  --jsonl_out  scanscribe_top5.jsonl
