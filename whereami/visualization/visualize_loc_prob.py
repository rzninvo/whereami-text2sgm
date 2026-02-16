#!/usr/bin/env python3
"""Dense-grid localization probability visualization.

For every ScanScribe caption graph:

1. Load its ground-truth 3D-SSG scene-graph + mesh.
2. Compute cosine similarity between all nodes; keep Top-K object matches.
3. Run the shared localization pipeline (grid sampling, ray-casting,
   probability computation, rendering).

This module is a thin wrapper that re-exports core utilities from
:mod:`whereami.localization` and provides its own ``main()`` for standalone
demo use.

Usage::

    python -m whereami.visualization.visualize_loc_prob \\
        --root $RSCAN_ROOT \\
        --graphs $WHEREAMI_DATA_ROOT/processed_data \\
        --top_k 5 --show_heatmap --show_3d
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from whereami.data_processing.scene_graph import SceneGraph

# Re-export all public helpers from whereami.localization so that
# existing code importing from this module keeps working.
from whereami.localization.grid import load_scene, sample_grid, first_hit_is_object  # noqa: F401
from whereami.localization.matching import topk_matched_objects  # noqa: F401
from whereami.localization.pipeline import run_loc_pipeline
from whereami.localization.visualization import (  # noqa: F401
    colour_objects,
    colormap,
    dir_to_yaw_pitch,
    best_fov_window,
    average_direction,
)


def main() -> None:
    """Standalone dense-grid localization demo over ScanScribe captions."""
    parser = argparse.ArgumentParser(
        description="Compute and visualise localisation probability surface "
                    "for ScanScribe captions, plus FOV-weighted arrow field.")
    parser.add_argument("--root", required=True,
                        help="Parent folder of 3RScan/<scan_id>/")
    parser.add_argument("--graphs", required=True,
                        help="processed_data/{3dssg,scanscribe}/")
    parser.add_argument("--top_k", type=int, default=25,
                        help="How many object matches to keep per caption")
    parser.add_argument("--grid_step", type=float, default=0.25,
                        help="XY grid spacing in metres")
    parser.add_argument("--query_limit", type=int,
                        help="Process only the first N captions (debug)")
    parser.add_argument("--show_heatmap", action="store_true",
                        help="Show 2-D Matplotlib heat-map")
    parser.add_argument("--show_3d", action="store_true",
                        help="Open Open3D viewer with mesh + probability spheres")
    parser.add_argument("--show_arrows", action="store_true",
                        help="Show FOV-weighted arrow (quiver) plot")
    parser.add_argument("--h_fov_deg", type=float, default=100.0,
                        help="Horizontal FOV in degrees")
    parser.add_argument("--v_fov_deg", type=float, default=60.0,
                        help="Vertical FOV in degrees")
    parser.add_argument("--arrow_stride", type=int, default=2,
                        help="Plot every Nth grid camera (reduce clutter)")
    parser.add_argument("--arrow_len", type=float, default=0.0,
                        help="Max arrow length in metres (0 = 0.9*grid_step)")
    args = parser.parse_args()

    print("\nConfiguration -------------------------------------------")
    for k, v in vars(args).items():
        print(f"  {k:<12}: {v}")
    print("---------------------------------------------------------\n")

    g3d = torch.load(Path(args.graphs) / "3dssg" /
                     "3dssg_graphs_processed_edgelists_relationembed.pt",
                     map_location="cpu")
    scenes = {sid: SceneGraph(sid,
                              graph_type="3dssg",
                              graph=g,
                              max_dist=1.0,
                              embedding_type="word2vec",
                              use_attributes=True)
              for sid, g in g3d.items()}

    gtxt = torch.load(Path(args.graphs) / "scanscribe" /
                      "scanscribe_text_graphs_from_image_desc_node_edge_features.pt",
                      map_location="cpu")
    queries = [SceneGraph(k.split("_")[0],
                          txt_id=None,
                          graph=g,
                          graph_type="scanscribe",
                          embedding_type="word2vec",
                          use_attributes=True)
               for k, g in gtxt.items()]
    if args.query_limit:
        queries = queries[: args.query_limit]

    for qi, qg in enumerate(queries, 1):
        sid = qg.scene_id
        sg = scenes[sid]

        obj_ids = topk_matched_objects(qg, sg, k=args.top_k)
        if not obj_ids:
            print(f"[{qi}] {sid} : no cosine matches — skipped")
            continue

        mesh, tri2obj, obj2faces = load_scene(Path(args.root) / sid)
        print(f"[{qi}] {sid}: {len(obj_ids)} matched objs")

        run_loc_pipeline(
            scan_dir=Path(args.root) / sid,
            obj_ids=obj_ids,
            obj2faces=obj2faces,
            mesh=mesh,
            tri2obj=tri2obj,
            grid_step=args.grid_step,
            show_heatmap=args.show_heatmap,
            show_arrows=args.show_arrows,
            show_3d=args.show_3d,
            h_fov_deg=args.h_fov_deg,
            v_fov_deg=args.v_fov_deg,
            arrow_stride=args.arrow_stride,
            arrow_len=args.arrow_len,
            title_prefix=f"{sid}  –  ",
        )


if __name__ == "__main__":
    main()
