#!/usr/bin/env python3
"""Open-set single-query inference: natural language to top-k 3DSSG scene matches.

Usage::

    python -m whereami.models.single_inference \
        --graphs $WHEREAMI_DATA_ROOT/processed_data \
        --ckpt $WHEREAMI_DATA_ROOT/model_checkpoints/graph2graph/best_model.pt \
        --query "There is a wooden chair next to a table." \
        --api_key_file /path/to/openai_key.txt \
        --top_k 5
"""

import argparse
import json

import numpy as np
import openai
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from whereami.data_processing.scene_graph import SceneGraph
from whereami.models.model_graph2graph import BigGNN
from whereami.models.inference import compute_match_score
from whereami.data_processing.create_text_embeddings import (
    create_embedding, create_embedding_clip, create_embedding_nlp
)


def embed_word(word: str, embedding_type="word2vec"):
    """Embeds a single word using the specified backend.

    Args:
        word: Word or phrase to embed.
        embedding_type: One of ``'word2vec'``, ``'clip'``, or ``'ada'``.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        ValueError: If ``embedding_type`` is not recognized.
    """
    if embedding_type == "word2vec":
        return create_embedding_nlp(word).tolist()
    elif embedding_type == "clip":
        return create_embedding_clip(word).tolist()
    elif embedding_type == "ada":
        return create_embedding(word)
    else:
        raise ValueError(f"Unknown embedding type {embedding_type}")


def parse_text_to_json(query_text: str, debug: bool = False) -> dict:
    """Uses GPT to extract a scene graph from a natural language description.

    Sends the query to GPT-4o-mini to parse objects, attributes, and
    relationships into a structured JSON graph.

    Args:
        query_text: Natural language scene description.
        debug: If True, prints raw LLM output and parsed JSON.

    Returns:
        Dictionary with ``'nodes'`` and ``'edges'`` lists ready for SceneGraph.

    Raises:
        ValueError: If the LLM returns invalid JSON that cannot be parsed.
    """
    client = openai.OpenAI(api_key=openai.api_key)
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


def text_to_scenegraph(query_text: str,
                       embedding_type="word2vec",
                       scene_id="query_0001", debug: bool = False):
    """Converts a natural language query into a SceneGraph.

    Parses the text with GPT, embeds all node labels, attributes, and edge
    relationships, then constructs a SceneGraph.

    Args:
        query_text: Natural language scene description.
        embedding_type: Embedding backend (``'word2vec'``, ``'clip'``, or ``'ada'``).
        scene_id: Scene ID to assign to the resulting graph.
        debug: If True, enables debug output during parsing.

    Returns:
        A SceneGraph constructed from the parsed and embedded text.
    """
    parsed = parse_text_to_json(query_text, debug)

    for node in parsed["nodes"]:
        node["label_" + embedding_type] = embed_word(node["label"], embedding_type)
        node["attributes_" + embedding_type] = {
            "all": [embed_word(a, embedding_type) for a in node["attributes"]]
        }

    for edge in parsed["edges"]:
        edge["relation_" + embedding_type] = embed_word(edge["relationship"], embedding_type)

    return SceneGraph(scene_id,
                      graph_type="scanscribe",
                      graph=parsed,
                      embedding_type=embedding_type,
                      use_attributes=True)


def parse_args():
    """Parses command-line arguments for single-query inference.

    Returns:
        Parsed argument namespace with graphs path, checkpoint, query text,
        embedding type, top_k, device, API key file, and debug flag.
    """
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
                   help="Path to file with line 'OPENAI_API_KEY=sk-...'", default=None)
    p.add_argument("--debug", action="store_true",
                   help="Enable debug mode (print LLM output, parsed graph, tqdm progress)")
    return p.parse_args()


def main():
    """Runs single-query text-to-scene retrieval against the 3DSSG database."""
    args = parse_args()

    # Load OpenAI API key
    with open(args.api_key_file, "r") as f:
        line = f.read().strip()
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1]
        else:
            key = line
        openai.api_key = key

    # Load 3DSSG database
    g3d_raw = torch.load(args.graphs / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt",
                         map_location="cpu", weights_only=False)
    database_3dssg = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=1.0, embedding_type=args.embedding_type,
                        use_attributes=True)
        for sid, g in g3d_raw.items()
    }

    # Load model
    model = BigGNN(N=1, heads=2).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False))
    model.eval()

    # Convert query text to SceneGraph
    query_graph = text_to_scenegraph(args.query,
                                     embedding_type=args.embedding_type,
                                     scene_id="query_0001", debug=args.debug)

    # Score against database
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
