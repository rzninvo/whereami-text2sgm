#!/bin/bash
# Helper script to run localization evaluation with sensible defaults.
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
  # localization.show_heatmap=true
  # localization.show_arrows=true
  # localization.show_3d=true
  localization.save_metrics="./eval/eval_metrics.json"
  localization.log_file="./eval/eval_loc_summary.log"
  localization.frame_policy=max_visible
  localization.top_k=10
  localization.grid_step=0.25
  localization.prediction_strategy=weighted
)

CMD=(
  python -m whereami.localization.cli
  localization.root="$SCENE_ROOT"
  localization.graphs="$GRAPHS_DIR"
  localization.query_root="$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  # Build a comma-separated list for Hydra list override
  IDS_STR=$(IFS=,; echo "${SCENE_IDS[*]}")
  CMD+=(localization.scene_ids="[$IDS_STR]")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
