"""Helper functions for scene graph analysis: similarity, loading, and subgraph matching."""

import torch
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import DBSCAN
import copy

from whereami.data_processing.scene_graph import SceneGraph

from whereami.utils.utils import _get_nlp


def np_cosine_sim(a, b):
    """Computes cosine similarity between two numpy vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score in [-1, 1].
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def nodes_features_similar(node1, node2, sim_thr=0.85):
    """Checks whether two nodes have similar feature vectors.

    Args:
        node1: First Node object.
        node2: Second Node object.
        sim_thr: Cosine similarity threshold.

    Returns:
        True if similarity exceeds the threshold.
    """
    return np_cosine_sim(node1.features, node2.features) > sim_thr


def load_scene_graphs(path):
    """Loads scene graphs from a ``.pt`` file.

    Args:
        path: Path to the serialized graph file.

    Returns:
        Dictionary of loaded scene graphs.
    """
    graphs = torch.load(path)
    return graphs


def load_text_graphs(path):
    """Loads text graphs from a ``.pt`` file.

    Args:
        path: Path to the serialized graph file.

    Returns:
        Dictionary of loaded text graphs.
    """
    graphs = torch.load(path)
    return graphs


def combine_node_features(graph1, graph2):
    """Concatenates node features from two graphs with graph-membership indices.

    Args:
        graph1: First SceneGraph.
        graph2: Second SceneGraph.

    Returns:
        Tuple of (all_node_features, graph_index) where graph_index is 0
        for graph1 nodes and 1 for graph2 nodes.
    """
    node_features1 = graph1.get_node_features()
    node_features2 = graph2.get_node_features()
    all_node_features = np.concatenate((node_features1, node_features2), axis=0)
    all_node_graph_index = np.concatenate((np.zeros(len(node_features1)), np.ones(len(node_features2))), axis=0)
    return all_node_features, all_node_graph_index


def get_matching_subgraph(graph1, graph2):
    """Extracts matching subgraphs from two scene graphs using DBSCAN clustering.

    Clusters all nodes from both graphs by feature similarity, then keeps
    only clusters containing nodes from both graphs to form subgraphs.

    Args:
        graph1: First SceneGraph (e.g. text query graph).
        graph2: Second SceneGraph (e.g. 3DSSG scene graph).

    Returns:
        Tuple of (subgraph1, subgraph2) as SceneGraph objects, or None
        if no matching nodes were found for a graph.
    """
    all_node_features, all_node_graph_index = combine_node_features(graph1, graph2)
    combined_node_idx = np.concatenate(([n1 for n1 in graph1.nodes], [n2 for n2 in graph2.nodes]), axis=0)
    assert all([i == graph1.nodes[i].idx for i in graph1.nodes]), \
        "graph1 node keys must equal node idx values"
    assert all([i == graph2.nodes[i].idx for i in graph2.nodes]), \
        "graph2 node keys must equal node idx values"
    idx_mapping = {}
    for i, idx in enumerate(combined_node_idx):
        idx_mapping[i] = idx

    clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(all_node_features)
    clusters = {}
    for i, cluster in enumerate(clustering.labels_):
        if cluster in clusters:
            clusters[cluster].append(i)
        else:
            clusters[cluster] = [i]

    graph1_keep_nodes = []
    graph2_keep_nodes = []
    for cluster in clusters:
        indices = clusters[cluster]
        graphs = [int(all_node_graph_index[i]) for i in indices]
        if 0 in graphs and 1 in graphs:
            graph1_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 0])
            graph2_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 1])

    assert isinstance(graph1, SceneGraph), "graph1 must be a SceneGraph"
    assert isinstance(graph2, SceneGraph), "graph2 must be a SceneGraph"
    graph1_keep_nodes = list(set(graph1_keep_nodes))
    graph2_keep_nodes = list(set(graph2_keep_nodes))
    subgraph1 = graph1.get_subgraph(graph1_keep_nodes, return_graph=True)
    subgraph2 = graph2.get_subgraph(graph2_keep_nodes, return_graph=True)

    return subgraph1, subgraph2


def calculate_overlap(graph1, graph2, sim_thr=0.95):
    """Calculates the fraction of graph1's nodes that have a similar match in graph2.

    Args:
        graph1: First SceneGraph.
        graph2: Second SceneGraph.
        sim_thr: Cosine similarity threshold for considering a node match.

    Returns:
        Overlap ratio in [0, 1], or 0 if either graph is None.
    """
    if graph1 is None or graph2 is None:
        return 0
    overlap = 0
    for node1id in graph1.nodes:
        node1 = graph1.nodes[node1id]
        found_match = False
        for node2id in graph2.nodes:
            node2 = graph2.nodes[node2id]
            if nodes_features_similar(node1, node2, sim_thr):
                found_match = True
                break
        if found_match:
            overlap += 1
    return overlap / len(graph1.nodes)
