#!/usr/bin/env python3
"""
inference_single.py
===================
Open-set inference:
Take a natural language query → build a text-graph → match against 3DSSG database.

Usage:
    python inference_single.py --graphs ../data_checkpoints/processed_data \
                               --ckpt ../model_checkpoints/graph2graph/best_model.pt \
                               --query "There is a wooden chair next to a table." \
                               --api_key_file ../keys/openai_key.txt \
                               --top_k 5
"""

import argparse, json, sys, torch, numpy as np, torch.nn.functional as F
import openai
from tqdm import tqdm
from pathlib import Path

# --------------------------------------------------------------------------- #
# Local imports                                                               #
# --------------------------------------------------------------------------- #
sys.path.append('../data_processing')
sys.path.append('../../../')

from scene_graph import SceneGraph
from model_graph2graph import BigGNN
from data_distribution_analysis.helper import get_matching_subgraph

from create_text_embeddings import create_embedding, create_embedding_clip, create_embedding_nlp


# --------------------------------------------------------------------------- #
# Helper: embed words depending on backend
# --------------------------------------------------------------------------- #
def embed_word(word: str, embedding_type="word2vec"):
    if embedding_type == "word2vec":
        # spaCy word2vec
        return create_embedding_nlp(word).tolist()
    elif embedding_type == "clip":
        return create_embedding_clip(word).tolist()
    elif embedding_type == "ada":
        return create_embedding(word)
    else:
        raise ValueError(f"Unknown embedding type {embedding_type}")


# --------------------------------------------------------------------------- #
# Helper: parse text to JSON for text graph formats using LLM (GPT-4o)
# --------------------------------------------------------------------------- #
def parse_text_to_json(query_text: str, debug: bool = False) -> dict:
    """
    Uses GPT to extract objects, attributes, and relationships from a text description.
    Returns a dict with "nodes" and "edges" ready for SceneGraph.
    """
    client = openai.OpenAI(api_key = openai.api_key)
    prompt = f"""
    You are a parser that converts natural language scene descriptions into a JSON graph.
    Extract:
    - objects (with id, label, attributes if any)
    - relationships (edges: source, target, relationship)

    Rules:
    - Assign each object an integer id starting at 0.
    - Each node: {{"id": int, "label": str, "attributes": [str,...]}}
    - Each edge: {{"source": int, "target": int, "relationship": str}}
    - If no attributes → "attributes": []
    - If no edges → "edges": []

    Example:
    Input: "There is a wooden chair next to a table."
    Output:
    {{
    "nodes": [
        {{"id": 0, "label": "chair", "attributes": ["wooden"]}},
        {{"id": 1, "label": "table", "attributes": []}}
    ],
    "edges": [
        {{"source": 0, "target": 1, "relationship": "next to"}}
    ]
    }}

    Now process:
    "{query_text}"
    Only output valid JSON, nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a JSON scene graph extractor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    if debug:
        print("\n[DEBUG] Raw LLM output:\n", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON substring if GPT adds extra text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise ValueError(f"LLM returned invalid JSON:\n{raw}")
        
    if debug:
        print("\n[DEBUG] Parsed JSON graph:")
        print(json.dumps(parsed, indent=2))

    return parsed


# --------------------------------------------------------------------------- #
# Convert query text into a SceneGraph
# --------------------------------------------------------------------------- #
def text_to_scenegraph(query_text: str,
                       embedding_type="word2vec",
                       scene_id="query_0001", debug: bool = False):
    parsed = parse_text_to_json(query_text, debug)

    # Embed nodes
    for node in parsed["nodes"]:
        node["label_" + embedding_type] = embed_word(node["label"], embedding_type)
        node["attributes_" + embedding_type] = {
            "all": [embed_word(a, embedding_type) for a in node["attributes"]]
        }

    # Embed edges
    for edge in parsed["edges"]:
        edge["relation_" + embedding_type] = embed_word(edge["relationship"], embedding_type)

    return SceneGraph(scene_id,
                      graph_type="scanscribe",
                      graph=parsed,
                      embedding_type=embedding_type,
                      use_attributes=True)


# --------------------------------------------------------------------------- #
# Compute similarity
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def compute_match_score(model: BigGNN | None,
                        qg: SceneGraph,
                        sg: SceneGraph,
                        device="cpu") -> float:
    q_sub, s_sub = get_matching_subgraph(qg, sg)
    def bad(g): return (g is None or len(g.nodes) <= 1
                        or (hasattr(g, "edge_idx") and len(g.edge_idx[0]) < 1))
    if bad(q_sub) or bad(s_sub):
        q_sub, s_sub = qg, sg

    def prep(g: SceneGraph):
        n, e, f = g.to_pyg()
        return (torch.tensor(np.array(n), dtype=torch.float32, device=device),
                torch.tensor(np.array(e[0:2]), dtype=torch.int64,   device=device),
                torch.tensor(np.array(f),      dtype=torch.float32, device=device))

    q_n, q_e, q_f = prep(q_sub)
    s_n, s_e, s_f = prep(s_sub)

    if model is None:
        cos = F.cosine_similarity(q_n.mean(0, keepdim=True),
                                  s_n.mean(0, keepdim=True), dim=1).item()
        return (cos + 1) / 2

    q_emb, s_emb, m_p = model(q_n, s_n, q_e, s_e, q_f, s_f)
    cos = (F.cosine_similarity(q_emb, s_emb, dim=0).item() + 1) / 2
    return 0.5 * m_p.item() + 0.5 * cos


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True, type=Path,
                   help="Folder containing processed_data/{3dssg}/ sub-folder")
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Trained BigGNN checkpoint (*.pt)")
    p.add_argument("--query", required=True, type=str,
                   help="Natural language query description")
    p.add_argument("--embedding_type", default="clip",
                   choices=["word2vec", "clip", "ada"])
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--api_key_file", type=Path,
                   help="Path to file with line 'OPENAI_API_KEY=sk-...'", default="/home/rohamzn/UZH Uni/Master Project/whereami-text2sgm/openai_api_key.txt")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug mode (print LLM output, parsed graph, tqdm progress)")
    return p.parse_args()


def main():
    args = parse_args()

    # 0) Load OpenAI API key
    with open(args.api_key_file, "r") as f:
        line = f.read().strip()
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1]
        else:
            key = line
        openai.api_key = key

    # 1) Load 3DSSG database
    g3d_raw = torch.load(args.graphs / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt",
                         map_location="cpu", weights_only=False)
    database_3dssg = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=1.0, embedding_type=args.embedding_type,
                        use_attributes=True)
        for sid, g in g3d_raw.items()
    }

    # 2) Load model
    model = BigGNN(N=1, heads=2).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False))
    model.eval()

    # 3) Convert query text → SceneGraph
    query_graph = text_to_scenegraph(args.query,
                                     embedding_type=args.embedding_type,
                                     scene_id="query_0001", debug=args.debug)

    # 4) Score against database
    scores = {}
    iterator = database_3dssg.items()

    if args.debug:
        iterator = tqdm(iterator, total=len(database_3dssg), desc="Scoring scenes")

    for sid, sg in iterator:
        scores[sid] = compute_match_score(model, query_graph, sg, args.device)

    best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.top_k]

    print(f"\nQuery: {args.query}")
    print("Top matches:")
    for rank, (sid, sc) in enumerate(best, 1):
        print(f"  {rank:>2}. {sid:<18}  score={sc:5.3f}")
    
    if args.debug:
        print("\n[DEBUG] Finished scoring all scenes.")


if __name__ == "__main__":
    main()
