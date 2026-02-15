#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

# Set these environment variables or edit the defaults below
SCENE_ROOT="${RSCAN_ROOT:?Set RSCAN_ROOT to your 3RScan data directory}"
GRAPHS_DIR="${WHEREAMI_DATA_ROOT:-./data}/processed_data"
CKPT_DIR="${WHEREAMI_DATA_ROOT:-./data}/model_checkpoints/graph2graph"

python3 -m whereami.visualization.visualization_minimal \
    --scan-id 095821f7-e2c2-2de1-9568-b9ce59920e29 \
    --root "$SCENE_ROOT" \
    --graphs "$GRAPHS_DIR" \
    --ckpt  "$CKPT_DIR/model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint.pt"
