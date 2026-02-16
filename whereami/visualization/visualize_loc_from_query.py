#!/usr/bin/env python3
"""Localise a natural-language query inside a specific 3RScan scene.

Bridges:
  - single_inference.py : builds a text SceneGraph from free text (LLM-backed)
  - localization pipeline : object matching, dense grid casting, visualisation

Usage::

    python -m whereami.visualization.visualize_loc_from_query \\
        --root /path/to/3RScan/data/3RScan \\
        --graphs /path/to/processed_data \\
        --scan_id 3RScan1234 \\
        --query "I can see a sofa facing a TV and a coffee table between them." \\
        --top_k 8 --grid_step 0.25 --show_heatmap --show_arrows --show_3d \\
        --h_fov_deg 100 --v_fov_deg 60 \\
        --api_key_file /path/to/openai_api_key.txt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from whereami.data_processing.scene_graph import SceneGraph
from whereami.localization.grid import load_scene
from whereami.localization.matching import topk_matched_objects
from whereami.localization.pipeline import run_loc_pipeline
from whereami.models.single_inference import text_to_scenegraph


def parse_args():
    p = argparse.ArgumentParser(
        description="Localise a custom natural-language query inside a specific 3RScan."
    )
    p.add_argument("--root", required=True,
                   help="Parent folder of 3RScan/<scan_id>/")
    p.add_argument("--graphs", required=True, type=Path,
                   help="processed_data folder containing 3dssg/*.pt")
    p.add_argument("--scan_id", required=True, type=str,
                   help="Target 3RScan scene ID (e.g., '3RScan1234')")
    p.add_argument("--query", required=True, type=str,
                   help="Natural language description to localise")

    p.add_argument("--top_k", type=int, default=25,
                   help="How many object matches to keep")
    p.add_argument("--grid_step", type=float, default=0.25,
                   help="XY grid spacing (m)")

    p.add_argument("--show_heatmap", action="store_true",
                   help="Show 2-D Matplotlib heatmap")
    p.add_argument("--show_3d", action="store_true",
                   help="Open Open3D viewer with mesh + probability spheres")
    p.add_argument("--show_arrows", action="store_true",
                   help="Show FOV-weighted arrow (quiver) plot")
    p.add_argument("--h_fov_deg", type=float, default=100.0,
                   help="Horizontal FOV in degrees")
    p.add_argument("--v_fov_deg", type=float, default=60.0,
                   help="Vertical FOV in degrees")
    p.add_argument("--arrow_stride", type=int, default=2,
                   help="Plot every Nth grid camera")
    p.add_argument("--arrow_len", type=float, default=0.0,
                   help="Max arrow length in metres (0 = 0.9*grid_step)")

    p.add_argument("--api_key_file", type=Path,
                   help="File containing OPENAI_API_KEY=sk-... or just the key")

    return p.parse_args()


def ensure_openai_key(api_key_file: Path | None):
    """Set openai.api_key from file or environment variable."""
    import openai
    if api_key_file is not None:
        text = Path(api_key_file).read_text().strip()
        key = text.split("=", 1)[1] if text.startswith("OPENAI_API_KEY=") else text
        openai.api_key = key
    else:
        if not (getattr(openai, "api_key", None) or os.getenv("OPENAI_API_KEY")):
            raise RuntimeError(
                "OpenAI API key not found. Pass --api_key_file or set OPENAI_API_KEY."
            )


def load_scene_graph_for_scan(graphs_dir: Path, scan_id: str) -> SceneGraph:
    """Load 3DSSG database and return SceneGraph for the requested scan_id."""
    g3d_path = graphs_dir / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt"
    if not g3d_path.exists():
        raise FileNotFoundError(g3d_path)

    g3d_all = torch.load(g3d_path, map_location="cpu", weights_only=False)
    if scan_id not in g3d_all:
        alts = [scan_id.replace("/", ""), scan_id.replace("3RScan/", ""), scan_id.split("/")[-1]]
        hit = next((a for a in alts if a in g3d_all), None)
        if hit is None:
            raise KeyError(f"scan_id '{scan_id}' not found in 3DSSG file.")
        scan_id = hit

    g = g3d_all[scan_id]
    sg = SceneGraph(scan_id,
                    graph_type="3dssg",
                    graph=g,
                    max_dist=1.0,
                    embedding_type="word2vec",
                    use_attributes=True)
    return sg


def main():
    args = parse_args()
    ensure_openai_key(args.api_key_file)

    # 1) Build a query SceneGraph from free text
    qg = text_to_scenegraph(args.query,
                            embedding_type="word2vec",
                            scene_id="query_0001",
                            debug=False)

    # 2) Load the target scene's 3DSSG
    sg = load_scene_graph_for_scan(args.graphs, args.scan_id)

    # 3) Top-K object matches
    obj_ids = topk_matched_objects(qg, sg, k=args.top_k)
    if not obj_ids:
        print("No cosine matches found between query and scene.")
        return

    # 4) Load mesh and run localization pipeline
    mesh, tri2obj, obj2faces = load_scene(Path(args.root) / sg.scene_id)
    print(f"[{sg.scene_id}] {len(obj_ids)} matched objs")

    run_loc_pipeline(
        scan_dir=Path(args.root) / sg.scene_id,
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
        title_prefix=f"{sg.scene_id} – ",
    )


if __name__ == "__main__":
    main()
