#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

python -m whereami.visualization.visualize_loc_from_query \
    scan_id=c7895f0b-339c-2d13-80e2-1d2aa04aa528 \
    inference.query="This is probably a wardrobe or storage room. I see many cabinets. There is a shelf attached to the wall holding a number of boxes." \
    localization.top_k=8 \
    localization.grid_step=0.25 \
    localization.show_heatmap=true \
    localization.show_arrows=true \
    localization.show_3d=true \
    localization.h_fov_deg=100 \
    localization.v_fov_deg=60
