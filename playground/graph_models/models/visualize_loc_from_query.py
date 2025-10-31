#!/usr/bin/env python3
"""
visualize_loc_from_query.py
---------------------------
Give me a scan ID (3RScan) and a *natural-language* description, and I’ll
visualise the probability of where the author stood when writing the query.

This bridges:
  • single_inference.py : builds a text SceneGraph from free text (LLM-backed)
  • visualize-loc-prob.py : does object matching, dense grid casting, viz

Example
-------
python visualize_loc_from_query.py \
  --root   /path/to/3RScan/data/3RScan \
  --graphs /path/to/whereami-text2sgm/playground/graph_models/processed_data \
  --scan_id 3RScan1234 \
  --query "I can see a sofa facing a TV and a coffee table between them." \
  --top_k 8 --grid_step 0.25 --show_heatmap --show_arrows --show_3d \
  --h_fov_deg 100 --v_fov_deg 60 \
  --api_key_file /path/to/openai_api_key.txt

Notes
-----
- Uses 'word2vec' embeddings to match the 3DSSG embeddings used by visualize-loc-prob.
- If --api_key_file is omitted, OPENAI_API_KEY env var will be used.
"""

from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import importlib.util
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c

# --------------------------------------------------------------------------- #
# Repo paths (same as the original scripts)
# --------------------------------------------------------------------------- #
sys.path.append('../data_processing')
sys.path.append('../../../')

from scene_graph import SceneGraph  # noqa: E402

# --------------------------------------------------------------------------- #
# Import functions from single_inference.py (clean module name)
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_SINGLE_INF = _THIS_DIR / "single_inference.py"
if not _SINGLE_INF.exists():
    raise FileNotFoundError(f"Expected single_inference.py at {_SINGLE_INF}")

spec = importlib.util.spec_from_file_location("single_inference", str(_SINGLE_INF))
single_inf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(single_inf)  # type: ignore

# text_to_scenegraph() is what we need
text_to_scenegraph = single_inf.text_to_scenegraph

# --------------------------------------------------------------------------- #
# Import functions from visualize_loc_prob.py (hyphenated filename)
# --------------------------------------------------------------------------- #
_VIZ = _THIS_DIR / "visualize_loc_prob.py"
if not _VIZ.exists():
    raise FileNotFoundError(f"Expected visualize_loc_prob.py at {_VIZ}")

vizmod = SourceFileLoader("vizlocprob", str(_VIZ)).load_module()  # type: ignore

 
# Pull in the helpers we'll reuse
load_scene             = vizmod.load_scene
topk_matched_objects   = vizmod.topk_matched_objects
sample_grid            = vizmod.sample_grid
first_hit_is_object    = vizmod.first_hit_is_object
colour_objects         = vizmod.colour_objects
colormap               = vizmod.colormap
dir_to_yaw_pitch       = getattr(vizmod, "dir_to_yaw_pitch", None)
best_fov_window        = getattr(vizmod, "best_fov_window", None)
average_direction      = getattr(vizmod, "average_direction", None)


# from visualize_loc_prob import load_scene, topk_matched_objects, sample_grid, first_hit_is_object, colour_objects, colormap
# from single_inference import text_to_scenegraph

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

    # Optional FOV-weighted arrows (if those helpers exist)
    p.add_argument("--show_arrows", action="store_true",
                   help="Show FOV-weighted arrow (quiver) plot")
    p.add_argument("--h_fov_deg", type=float, default=60.0,
                   help="Horizontal FOV in degrees")
    p.add_argument("--v_fov_deg", type=float, default=100.0,
                   help="Vertical FOV in degrees")
    p.add_argument("--arrow_stride", type=int, default=2,
                   help="Plot every Nth grid camera")
    p.add_argument("--arrow_len", type=float, default=0.0,
                   help="Max arrow length in metres (0 → 0.9*grid_step)")

    # OpenAI key for the LLM parser in single_inference
    p.add_argument("--api_key_file", type=Path,
                   help="File containing OPENAI_API_KEY=sk-... or just the key")

    return p.parse_args()


def ensure_openai_key(api_key_file: Path | None):
    # single_inference.text_to_scenegraph relies on openai.api_key being set.
    import openai
    if api_key_file is not None:
        text = Path(api_key_file).read_text().strip()
        key = text.split("=", 1)[1] if text.startswith("OPENAI_API_KEY=") else text
        openai.api_key = key
    else:
        # Use environment if present
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
        # Try common variants
        alts = [scan_id.replace("/", ""), scan_id.replace("3RScan/", ""), scan_id.split("/")[-1]]
        hit = next((a for a in alts if a in g3d_all), None)
        if hit is None:
            raise KeyError(f"scan_id '{scan_id}' not found in 3DSSG file.")
        scan_id = hit

    g = g3d_all[scan_id]
    # Match embedding_type used by visualize-loc-prob’s matcher
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

    # 1) Build a query SceneGraph from free text (word2vec to match 3DSSG)
    qg = text_to_scenegraph(args.query,
                            embedding_type="word2vec",
                            scene_id="query_0001",
                            debug=False)

    # 2) Load the target scene’s 3DSSG
    sg = load_scene_graph_for_scan(args.graphs, args.scan_id)

    # 3) Top-K object matches in this scene
    obj_ids = topk_matched_objects(qg, sg, k=args.top_k)
    if not obj_ids:
        print("No cosine matches found between query and scene.")
        return

    # 4) Load mesh & raycaster
    mesh, tri2obj, obj2faces = load_scene(Path(args.root) / sg.scene_id)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    verts = np.asarray(mesh.vertices)
    cams  = sample_grid(verts, step=args.grid_step)

    # Recover grid layout (Nx, Ny) for optional arrow stride logic
    xs, ys = verts[:, 0], verts[:, 1]
    gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
    Nx, Ny = len(gx), len(gy)

    print(f"[{sg.scene_id}] grid {len(cams):,} pts  |  {len(obj_ids)} matched objs")

    # 5) Object centroids
    tris = np.asarray(mesh.triangles)
    centroids = {}
    for oid in obj_ids:
        faces = obj2faces.get(oid)
        if faces is not None and len(faces):
            centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)

    if not centroids:
        print("No centroids available for matched objects — aborting.")
        return

    # 6) Visibility tally per candidate camera (probability surface)
    visible_dirs = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, cen in centroids.items():
            if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                d = cen - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)

    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    if counts.sum() == 0:
        print("Matched objects are not visible from any grid camera.")
        return
    probs = counts / counts.sum()

    # 7) 2-D heatmap
    if args.show_heatmap:
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                         cmap="viridis", s=12)
        plt.colorbar(sc, label="probability")
        plt.title(f"{sg.scene_id} — grid {args.grid_step} m")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    # 8) Optional: FOV-weighted arrows (if helpers exist)
    if args.show_arrows and dir_to_yaw_pitch and best_fov_window and average_direction:
        hfov = math.radians(args.h_fov_deg)
        vfov = math.radians(args.v_fov_deg)
        max_len = (0.9 * args.grid_step) if args.arrow_len <= 0 else args.arrow_len

        Qx, Qy, U, V, W = [], [], [], [], []
        stride = max(1, int(args.arrow_stride))

        for gy_i in range(0, Ny, stride):
            for gx_i in range(0, Nx, stride):
                idx = gy_i * Nx + gx_i
                dirs = np.asarray(visible_dirs[idx], dtype=np.float32)
                if dirs.size == 0:
                    continue

                yaws = np.empty(len(dirs), dtype=np.float32)
                pits = np.empty(len(dirs), dtype=np.float32)
                for i, v in enumerate(dirs):
                    y, p = dir_to_yaw_pitch(v)
                    yaws[i] = y; pits[i] = p

                sel, count = best_fov_window(yaws, pits, hfov, vfov)
                if count == 0:
                    continue

                mdir = average_direction(dirs, sel)
                if mdir is None:
                    continue

                xy = mdir[:2]
                nxy = np.linalg.norm(xy)
                if nxy < 1e-8:
                    continue
                xy_unit = xy / nxy

                Qx.append(cams[idx, 0]); Qy.append(cams[idx, 1])
                U.append(float(xy_unit[0])); V.append(float(xy_unit[1])); W.append(count)

        if len(W):
            W = np.array(W, dtype=np.float32)
            scale_fac = np.where(W > 0, W / W.max(), 0.0)
            U = np.array(U) * max_len * scale_fac
            V = np.array(V) * max_len * scale_fac

            plt.figure(figsize=(7, 7))
            plt.quiver(Qx, Qy, U, V, W, angles="xy", scale_units="xy",
                       scale=1.0, cmap="viridis", width=0.004, minlength=0.01)
            plt.colorbar(label="max visible objects within FOV")
            plt.title(f"{sg.scene_id} – FOV-weighted directions "
                      f"(H={args.h_fov_deg}°, V={args.v_fov_deg}°, stride={stride})")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()
        else:
            print("Arrows: no valid FOV windows to plot.")

    # 9) 3-D viewer
    if args.show_3d:
        vis_mesh = colour_objects(mesh, obj2faces, list(centroids.keys()))
        spheres  = []
        for p, col in zip(cams, colormap(probs)):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            s.translate(p);  s.paint_uniform_color(col);  spheres.append(s)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=800,
                          window_name=f"{sg.scene_id} – localisation prob.")
        vis.add_geometry(vis_mesh)
        for s in spheres:
            vis.add_geometry(s)
        vis.get_render_option().point_size = 3
        vis.run();  vis.destroy_window()


if __name__ == "__main__":
    main()
