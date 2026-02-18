#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

python -m whereami.models.inference \
    eval.model_name=model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint \
    inference.top_k=5 \
    inference.jsonl_out=scanscribe_top5.jsonl \
    device=cuda
