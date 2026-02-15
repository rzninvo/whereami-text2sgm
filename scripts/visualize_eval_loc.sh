#!/bin/bash
# Helper script to run visualize_eval_loc.py with sensible defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

# 3RScan mesh root (scene folders with meshes + instance labels)
SCENE_ROOT="${RSCAN_ROOT:?Set RSCAN_ROOT to your 3RScan data directory}"

# Caption JSON root (frame-*.json files with ground-truth poses & visible objects)
QUERY_ROOT="${WHEREAMI_QUERY_ROOT:-./datasets/3RScan_processed}"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="${WHEREAMI_DATA_ROOT:-./data}/processed_data"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
SCENE_IDS=()

# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  # --show_heatmap
  # --show_arrows
  # --show_3d
  --save_metrics "./eval/eval_metrics.json"
  --log_file "./eval/eval_loc_summary.log"
  --frame_policy max_visible
  --top_k 10
  --grid_step 0.25
  --prediction_strategy "weighted"
)

CMD=(
  python -m whereami.visualization.visualize_eval_loc
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
