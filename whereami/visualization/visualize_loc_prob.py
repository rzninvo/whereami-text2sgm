#!/usr/bin/env python3
"""Dense-grid localization probability visualization.

For every ScanScribe caption graph:

1. Load its ground-truth 3D-SSG scene-graph + mesh.
2. Compute cosine similarity between all nodes; keep Top-K object matches.
3. Generate a dense XY grid (spacing = ``--grid_step``) at eye-height.
4. For each grid cell (candidate camera) cast one ray to every matched
   object centroid; count how many objects are the *first* hit.
5. Convert counts to posterior probabilities.
6. Visualise:

   - probability heat-map (optional, 2-D)
   - FOV-weighted arrow (quiver) plot (optional, 2-D)
   - full 3-D mesh with matched objects bright and camera spheres whose
     colour encodes probability (optional, Open3D).

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
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from whereami.data_processing.scene_graph import SceneGraph

# Re-export all public helpers from whereami.localization so that
# existing code importing from this module keeps working.
from whereami.localization.grid import load_scene, sample_grid, first_hit_is_object  # noqa: F401
from whereami.localization.matching import topk_matched_objects  # noqa: F401
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
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        verts = np.asarray(mesh.vertices)
        cams = sample_grid(verts, step=args.grid_step)
        xs, ys = verts[:, 0], verts[:, 1]
        gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
        gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
        Nx, Ny = len(gx), len(gy)

        print(f"[{qi}] {sid}: grid {len(cams):,} pts  |  {len(obj_ids)} objs")

        tris = np.asarray(mesh.triangles)
        centroids = {}
        for oid in obj_ids:
            faces = obj2faces.get(oid)
            if faces is not None and len(faces):
                centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)

        if not centroids:
            print("    no centroids for matched objects — skipped\n")
            continue

        visible_dirs = [[] for _ in range(len(cams))]
        for idx, cam in enumerate(cams):
            for oid, cen in centroids.items():
                if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                    d = cen - cam
                    ln = np.linalg.norm(d)
                    if ln > 1e-6:
                        visible_dirs[idx].append(d / ln)

        scores = np.array([len(v) for v in visible_dirs], dtype=np.int32)
        if scores.sum() == 0:
            print("    none of the matched objects visible — skipped\n")
            continue
        probs = scores / scores.sum()

        if args.show_heatmap:
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                             cmap="viridis", s=12)
            plt.colorbar(sc, label="probability")
            plt.title(f"{sid}  –  grid {args.grid_step} m")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        if args.show_arrows:
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
                        yaws[i] = y
                        pits[i] = p
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
                    Qx.append(cams[idx, 0])
                    Qy.append(cams[idx, 1])
                    U.append(float(xy_unit[0]))
                    V.append(float(xy_unit[1]))
                    W.append(count)

            if len(W) == 0:
                print("    arrows: no valid FOV windows — nothing to plot")
            else:
                W = np.array(W, dtype=np.float32)
                scale_fac = np.where(W > 0, W / W.max(), 0.0)
                U = np.array(U) * max_len * scale_fac
                V = np.array(V) * max_len * scale_fac
                plt.figure(figsize=(7, 7))
                plt.quiver(Qx, Qy, U, V, W, angles="xy", scale_units="xy",
                           scale=1.0, cmap="viridis", width=0.004,
                           minlength=0.01)
                plt.colorbar(label="max visible objects within FOV")
                plt.title(f"{sid} – FOV-weighted average directions "
                          f"(H={args.h_fov_deg}°, V={args.v_fov_deg}°, stride={stride})")
                plt.axis("equal")
                plt.tight_layout()
                plt.show()

        if args.show_3d:
            vis_mesh = colour_objects(mesh, obj2faces, obj_ids)
            spheres = []
            for p, col in zip(cams, colormap(probs)):
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                s.translate(p)
                s.paint_uniform_color(col)
                spheres.append(s)

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=800,
                              window_name=f"{sid} – localisation prob.")
            vis.add_geometry(vis_mesh)
            for s in spheres:
                vis.add_geometry(s)
            vis.get_render_option().point_size = 3
            vis.run()
            vis.destroy_window()


if __name__ == "__main__":
    main()
