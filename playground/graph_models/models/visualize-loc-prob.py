#!/usr/bin/env python3
"""
visualize-loc-prob.py  ·  dense-grid version  ·  June 2025
-----------------------------------------------------------

For every ScanScribe caption graph:

1.  Load its ground-truth 3D-SSG scene-graph + mesh.
2.  Compute cosine similarity between all nodes → keep Top-K object matches.
3.  Generate a dense XY grid (spacing = --grid_step) at eye-height.
4.  For each grid cell (≙ candidate camera) cast one ray to every matched
    object centroid; count how many objects are the *first* hit.
5.  Convert counts → posterior probabilities.
6.  Visualise
      • probability heat-map (optional, 2-D)  
      • full 3-D mesh with matched objects bright and camera spheres whose
        colour encodes probability (optional, Open3D).



Adds a second 2-D plot: a quiver of *weighted arrows* whose:
  • direction = average of casted ray directions (unit vectors) that fit in a
    camera FOV window (H×V) which *maximizes* the number of visible objects.
  • size     = that maximal count (scaled).
                

Author: Shak

python visualize-loc-prob.py     --root  /home/klrshak/work/VisionLang/3RScan/data/3RScan     --graphs /home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data --top_k 5 --show_heatmap --show_3d

New CLI flags:
  --show_arrows         Show weighted-arrow (quiver) plot
  --h_fov_deg 100       Horizontal FOV (degrees)
  --v_fov_deg 60        Vertical FOV (degrees)
  --arrow_stride 2      Plot every Nth grid camera (reduce clutter)
  --arrow_len 0.0       Max arrow length in metres (0 → 0.9*grid_step)

The rest of the pipeline (heatmap, Open3D) is unchanged.
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c

# --------------------------------------------------------------------------- #
#  Repo imports                                                               #
# --------------------------------------------------------------------------- #

sys.path.append('../data_processing')
sys.path.append('../../../')

from scene_graph import SceneGraph                    # noqa: E402
from data_distribution_analysis.helper import get_matching_subgraph  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
#  Utility: load mesh + obj↔face maps                                         ═
# ════════════════════════════════════════════════════════════════════════════
def load_scene(scan_dir: Path):
    """Return (legacy mesh, faces→object-id array, obj→faces dict)."""
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    vc   = (np.asarray(mesh.vertex_colors) * 255 + 0.5).astype(np.uint32)
    vhex = (vc[:, 0] << 16) | (vc[:, 1] << 8) | vc[:, 2]

    meta = {
        s["scan"]: s
        for s in json.load(open(scan_dir.parent / "objects.json"))["scans"]
    }[scan_dir.name]
    color2oid = {int(o["ply_color"].lstrip("#"), 16): int(o["id"])
                 for o in meta["objects"]}

    v_oid = np.array([color2oid.get(int(h), 0) for h in vhex], dtype=np.int32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array([np.bincount(v_oid[t]).argmax() for t in tris],
                       dtype=np.int32)

    obj2faces = {}
    for fid, oid in enumerate(tri2obj):
        if oid != 0:
            obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}
    return mesh, tri2obj, obj2faces


# ════════════════════════════════════════════════════════════════════════════
#  Cosine-similarity Top-K matcher                                            ═
# ════════════════════════════════════════════════════════════════════════════
def topk_matched_objects(qg: SceneGraph, sg: SceneGraph, k: int = 5):
    qf, _, _ = qg.to_pyg()
    sf, _, _ = sg.to_pyg()
    qf = F.normalize(torch.tensor(np.asarray(qf), dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(np.asarray(sf), dtype=torch.float32), dim=1)

    sim = qf @ sf.T                                   # (|Q|, |S|)
    topv, topi = torch.topk(sim.flatten(), min(k, sim.numel()))
    sids  = list(sg.nodes)
    S     = sf.size(0)
    picks = []
    for idx in topi.tolist():
        sid = sids[idx % S]
        if sid not in picks:
            picks.append(sid)
        if len(picks) == k:
            break
    return picks


# ════════════════════════════════════════════════════════════════════════════
#  Camera grid sampler                                                        ═
# ════════════════════════════════════════════════════════════════════════════
def sample_grid(verts: np.ndarray, step: float, z_eye: float = 1.6):
    """
    Return (N,3) xyz grid points covering the mesh’s XY AABB, spacing = step.
    """
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    gx = np.arange(xs.min(), xs.max() + 1e-4, step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, step)
    xv, yv = np.meshgrid(gx, gy, indexing="xy")
    n = xv.size
    cams = np.stack([xv.ravel(), yv.ravel(), np.full(n, zs.min() + z_eye)],
                    axis=1)
    return cams


# ════════════════════════════════════════════════════════════════════════════
#  Ray-visibility test (single ray)                                           ═
# ════════════════════════════════════════════════════════════════════════════
def first_hit_is_object(cam: np.ndarray, centre: np.ndarray, target_oid: int,
                        rc: o3d.t.geometry.RaycastingScene,
                        tri2obj: np.ndarray) -> bool:
    d = centre - cam
    l = np.linalg.norm(d)
    if l < 1e-6:
        return False
    ray = np.concatenate([cam, d / l])[None, :]
    ans = rc.cast_rays(o3c.Tensor(ray, dtype=o3c.Dtype.Float32))
    tri = int(ans["primitive_ids"].cpu().numpy()[0])
    if tri < 0 or tri >= len(tri2obj):
        return False
    return int(tri2obj[tri]) == int(target_oid)


# ════════════════════════════════════════════════════════════════════════════
#  Colour helpers                                                             ═
# ════════════════════════════════════════════════════════════════════════════
def colour_objects(mesh: o3d.geometry.TriangleMesh,
                   obj2faces: dict[int, np.ndarray],
                   focus: list[int]):
    """Grey base mesh; bright random colour for every object in `focus`."""
    rng = np.random.default_rng(42)
    grey = np.full((len(mesh.vertices), 3), 0.55)
    tris = np.asarray(mesh.triangles)
    for oid in focus:
        for fid in obj2faces.get(oid, []):
            for vid in tris[fid]:
                grey[int(vid)] = rng.random(3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(grey)
    return mesh


def colormap(vals: np.ndarray):
    """Map values in [0,1] → RGB using matplotlib’s 'hot'."""
    cmap = plt.get_cmap("viridis")
    return cmap(vals)[:, :3]


# ════════════════════════════════════════════════════════════════════════════
#  New: angular utilities + FOV maximisation                                  ═
# ════════════════════════════════════════════════════════════════════════════
def dir_to_yaw_pitch(v: np.ndarray):
    """Return yaw (around +Z, from +X toward +Y) and pitch (up) in radians."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    yaw = math.atan2(y, x)
    pitch = math.atan2(z, math.hypot(x, y))
    return yaw, pitch


def best_fov_window(yaws: np.ndarray,
                    pitches: np.ndarray,
                    hfov: float, vfov: float):
    """
    Given arrays of angles (rad), find subset that maximises count inside an
    axis-aligned yaw×pitch window of size hfov×vfov. Handles yaw wrap-around.
    Returns (selected_indices, max_count).
    """
    n = len(yaws)
    if n == 0:
        return np.array([], dtype=int), 0

    order = np.argsort(yaws)
    yaw_sorted = yaws[order]
    pit_sorted = pitches[order]
    idx_sorted = order

    yaw_ext = np.concatenate([yaw_sorted, yaw_sorted + 2 * math.pi])
    pit_ext = np.concatenate([pit_sorted, pit_sorted])
    idx_ext = np.concatenate([idx_sorted, idx_sorted])

    best_cnt = 0
    best_sel = np.array([], dtype=int)

    j = 0
    for s in range(n):
        y0 = yaw_ext[s]
        y1 = y0 + hfov
        j = max(j, s)
        while j < s + n and yaw_ext[j] <= y1 + 1e-9:
            j += 1

        if j <= s:
            continue

        # candidates within horizontal window
        cand_slice = slice(s, j)
        ps = pit_ext[cand_slice]
        ids = idx_ext[cand_slice]

        # 1D sliding window on pitch
        p_order = np.argsort(ps)
        ps = ps[p_order]
        ids = ids[p_order]

        t_end = 0
        for t_start in range(len(ps)):
            p0 = ps[t_start]
            p1 = p0 + vfov
            while t_end < len(ps) and ps[t_end] <= p1 + 1e-9:
                t_end += 1
            cnt = t_end - t_start
            if cnt > best_cnt:
                best_cnt = cnt
                best_sel = np.unique(ids[t_start:t_end])

        # (implicit: move to next s)

    return best_sel, int(best_cnt)


def average_direction(unit_dirs: np.ndarray, sel: np.ndarray):
    """Vector-average the selected unit directions; return unit 3D vector or None."""
    if sel.size == 0:
        return None
    m = unit_dirs[sel].mean(axis=0)
    n = np.linalg.norm(m)
    if n < 1e-8:
        return None
    return m / n


# ════════════════════════════════════════════════════════════════════════════
#  Main programme                                                             ═
# ════════════════════════════════════════════════════════════════════════════
def main():
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

    # New: arrows / FOV options
    parser.add_argument("--show_arrows", action="store_true",
                        help="Show FOV-weighted arrow (quiver) plot")
    parser.add_argument("--h_fov_deg", type=float, default=100.0,
                        help="Horizontal FOV in degrees")
    parser.add_argument("--v_fov_deg", type=float, default=60.0,
                        help="Vertical FOV in degrees")
    parser.add_argument("--arrow_stride", type=int, default=2,
                        help="Plot every Nth grid camera (reduce clutter)")
    parser.add_argument("--arrow_len", type=float, default=0.0,
                        help="Max arrow length in metres (0 → 0.9*grid_step)")

    args = parser.parse_args()

    # ----- quick summary of chosen arguments
    print("\nConfiguration -------------------------------------------")
    for k, v in vars(args).items():
        print(f"  {k:<12}: {v}")
    print("---------------------------------------------------------\n")

    # ----- load scene graphs ----------------------------------------------
    g3d = torch.load(Path(args.graphs) / "3dssg" /
                     "3dssg_graphs_processed_edgelists_relationembed.pt",
                     map_location="cpu")
    scenes = {sid: SceneGraph(sid,
                              graph_type="3dssg",
                              graph=g,
                              max_dist=1.0,        # ← FIX
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

    # ----- iterate over captions ------------------------------------------
    for qi, qg in enumerate(queries, 1):
        sid = qg.scene_id
        sg  = scenes[sid]

        obj_ids = topk_matched_objects(qg, sg, k=args.top_k)
        if not obj_ids:
            print(f"[{qi}] {sid} : no cosine matches — skipped")
            continue

        # mesh + ray-caster
        mesh, tri2obj, obj2faces = load_scene(Path(args.root) / sid)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        verts = np.asarray(mesh.vertices)
        cams  = sample_grid(verts, step=args.grid_step)
        # Recompute gx/gy to know grid dimensions (match sample_grid logic)
        xs, ys = verts[:, 0], verts[:, 1]
        gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
        gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
        Nx, Ny = len(gx), len(gy)

        print(f"[{qi}] {sid}: grid {len(cams):,} pts  |  {len(obj_ids)} objs")

        # object centroids
        tris = np.asarray(mesh.triangles)
        centroids = {}
        for oid in obj_ids:
            faces = obj2faces.get(oid)
            if faces is not None and len(faces):
                centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)

        if not centroids:
            print("    no centroids for matched objects — skipped\n")
            continue

        # ---- visibility + per-cam direction list (reuses single ray cast)
        visible_dirs = [[] for _ in range(len(cams))]
        for idx, cam in enumerate(cams):
            for oid, cen in centroids.items():
                if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                    d = cen - cam
                    l = np.linalg.norm(d)
                    if l > 1e-6:
                        visible_dirs[idx].append(d / l)

        scores = np.array([len(v) for v in visible_dirs], dtype=np.int32)
        if scores.sum() == 0:
            print("    none of the matched objects visible — skipped\n")
            continue
        probs = scores / scores.sum()

        # --- 2-D heat-map
        if args.show_heatmap:
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                             cmap="viridis", s=12)
            plt.colorbar(sc, label="probability")
            plt.title(f"{sid}  –  grid {args.grid_step} m")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        # --- New: weighted arrow plot (quiver) -----------------------------
        if args.show_arrows:
            hfov = math.radians(args.h_fov_deg)
            vfov = math.radians(args.v_fov_deg)
            max_len = (0.9 * args.grid_step) if args.arrow_len <= 0 else args.arrow_len

            # Prepare arrays for quiver
            Qx, Qy, U, V, W = [], [], [], [], []  # positions, vectors, weights
            max_score = 0

            # Iterate subset of grid in a structured way using Nx,Ny
            stride = max(1, int(args.arrow_stride))
            for gy_i in range(0, Ny, stride):
                for gx_i in range(0, Nx, stride):
                    idx = gy_i * Nx + gx_i
                    dirs = np.asarray(visible_dirs[idx], dtype=np.float32)
                    if dirs.size == 0:
                        continue

                    # Yaw/pitch arrays
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

                    # Project to XY for arrow direction
                    xy = mdir[:2]
                    nxy = np.linalg.norm(xy)
                    if nxy < 1e-8:
                        continue
                    xy_unit = xy / nxy

                    max_score = max(max_score, count)
                    Qx.append(cams[idx, 0])
                    Qy.append(cams[idx, 1])
                    U.append(float(xy_unit[0]))   # scaled later
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

        # --- 3-D viewer
        if args.show_3d:
            vis_mesh = colour_objects(mesh, obj2faces, obj_ids)
            spheres  = []
            for p, col in zip(cams, colormap(probs)):
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                s.translate(p);  s.paint_uniform_color(col);  spheres.append(s)

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=800,
                              window_name=f"{sid} – localisation prob.")
            vis.add_geometry(vis_mesh)
            for s in spheres:
                vis.add_geometry(s)
            vis.get_render_option().point_size = 3
            vis.run();  vis.destroy_window()


if __name__ == "__main__":
    main()
