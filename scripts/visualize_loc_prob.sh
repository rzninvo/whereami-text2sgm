#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

SCENE_ROOT="${RSCAN_ROOT:?Set RSCAN_ROOT to your 3RScan data directory}"
GRAPHS_DIR="${WHEREAMI_DATA_ROOT:-./data}/processed_data"

python -m whereami.visualization.visualize_loc_from_query \
  --root "$SCENE_ROOT" \
  --graphs "$GRAPHS_DIR" \
  --scan_id c7895f0b-339c-2d13-80e2-1d2aa04aa528 \
  --query "This is probably a wardrobe or storage room. I see many cabinets. There is a shelf attached to the wall holding a number of boxes." \
  --top_k 8 --grid_step 0.25 --show_heatmap --show_arrows --show_3d \
  --h_fov_deg 100 --v_fov_deg 60
