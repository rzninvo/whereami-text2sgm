#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

python3 -m whereami.visualization.visualization_minimal \
    scan_id=095821f7-e2c2-2de1-9568-b9ce59920e29 \
    eval.model_name=model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint
