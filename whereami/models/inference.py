#!/usr/bin/env python3
"""Batch text-to-scene retrieval: ScanScribe caption graph to top-k matching 3D-SSG scenes.

Scoring: 0.5 * matching-probability + 0.5 * cosine-similarity (both in [0,1]).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from whereami.data_processing.scene_graph import SceneGraph
from whereami.analysis.helper import get_matching_subgraph
from whereami.models.model_graph2graph import BigGNN


@torch.inference_mode()
def compute_match_score(model: BigGNN | None,
                        qg: SceneGraph,
                        sg: SceneGraph,
                        device: str = "cpu") -> float:
    """Computes a blended matching score between a query graph and a scene graph.

    Extracts matching subgraphs, converts to PyG format, runs through the model
    (or falls back to cosine-only), and returns a score in [0, 1].

    Args:
        model: Trained BigGNN model, or None for cosine-only scoring.
        qg: Query (text) scene graph.
        sg: Database (3DSSG) scene graph.
        device: Torch device string.

    Returns:
        Blended score in [0, 1]: 0.5 * matching_prob + 0.5 * cosine_sim.
    """
    q_sub, s_sub = get_matching_subgraph(qg, sg)

    def bad(g):
        return (g is None or len(g.nodes) <= 1
                or (hasattr(g, "edge_idx") and len(g.edge_idx[0]) < 1))

    if bad(q_sub) or bad(s_sub):
        q_sub, s_sub = qg, sg

    def prep(g: SceneGraph):
        n, e, f = g.to_pyg()
        return (torch.tensor(np.array(n), dtype=torch.float32, device=device),
                torch.tensor(np.array(e[0:2]), dtype=torch.int64, device=device),
                torch.tensor(np.array(f), dtype=torch.float32, device=device))

    q_n, q_e, q_f = prep(q_sub)
    s_n, s_e, s_f = prep(s_sub)

    if model is None:
        cos = F.cosine_similarity(q_n.mean(0, keepdim=True),
                                  s_n.mean(0, keepdim=True), dim=1).item()
        return (cos + 1) / 2

    q_emb, s_emb, m_p = model(q_n, s_n, q_e, s_e, q_f, s_f)
    cos = (F.cosine_similarity(q_emb, s_emb, dim=0).item() + 1) / 2
    return 0.5 * m_p.item() + 0.5 * cos


def parse_args():
    """Parses command-line arguments for batch inference.

    Returns:
        Parsed argument namespace with graphs path, checkpoint, top_k, device,
        and optional JSONL output path.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True, type=Path,
                   help="Folder that contains the processed_data/{3dssg,scanscribe}/ sub-folders")
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Trained BigGNN checkpoint (*.pt)")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jsonl_out", type=Path,
                   help="Write one ranked-list per query to this JSONL file")
    return p.parse_args()


def main():
    """Runs batch text-to-scene retrieval over all ScanScribe captions."""
    args = parse_args()
    t0 = time.perf_counter()

    # Load graphs
    g3d_raw = torch.load(args.graphs / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt",
                         map_location="cpu")
    scans_raw = torch.load(args.graphs / "scanscribe" / "scanscribe_text_graphs_from_image_desc_node_edge_features.pt",
                           map_location="cpu")

    database_3dssg = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=1.0, embedding_type="word2vec",
                        use_attributes=True)
        for sid, g in g3d_raw.items()
    }
    queries = [
        SceneGraph(k.split("_")[0], txt_id=None,
                   graph=g, graph_type="scanscribe",
                   embedding_type="word2vec", use_attributes=True)
        for k, g in scans_raw.items()
    ]
    print(f"Loaded {len(queries)} ScanScribe captions, "
          f"{len(database_3dssg)} 3D-SSG scenes.")

    # Load model
    device = args.device
    model = BigGNN(N=1, heads=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # For each caption, rank all scenes
    jsonl = None
    if args.jsonl_out:
        jsonl = open(args.jsonl_out, "w")

    for qi, qg in enumerate(queries, 1):
        scores = {
            sid: compute_match_score(model, qg, sg, device)
            for sid, sg in database_3dssg.items()
        }
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.top_k]

        print(f"\nQuery {qi:>4}/{len(queries)}  (scene_id={qg.scene_id})")
        for rank, (sid, sc) in enumerate(best, 1):
            gt_tag = "  *GT*" if sid == qg.scene_id else ""
            print(f"  {rank:>2}. {sid:<18}  score={sc:5.3f}{gt_tag}")

        if jsonl:
            jsonl.write(json.dumps({
                "query_scene_id": qg.scene_id,
                "top_k": best
            }) + "\n")

    if jsonl:
        jsonl.close()
        print(f"\nWrote ranked lists to {args.jsonl_out}")

    print(f"\nFinished in {(time.perf_counter()-t0):.1f}s.")


if __name__ == "__main__":
    main()
