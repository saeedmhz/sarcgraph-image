import copy
import numpy as np
import networkx as nx

from sarcomere_detection.graph_operations import (
    create_knn_graph,
    get_angle,
    update_scores,
    graph_pruning,
)


def score_graph(G, config, _sarc_vector, _sarc_score):
    c_avg_length = config.coeff_avg_length
    l_avg = config.avg_sarc_length
    l_max = config.max_sarc_length
    l_min = config.min_sarc_length
    edges_attr_dict = {}
    for node in range(G.number_of_nodes()):
        for neighbor in G.neighbors(node):
            score = 0
            v1, l1 = _sarc_vector(G, node, neighbor)
            if l1 <= l_max and l1 >= l_min:
                d = min(l_avg - l_min, l_max - l_avg)
                avg_length_score = np.exp(-np.pi * (abs(l1 - l_avg) / d) ** 4)
                for far_neighbor in G.neighbors(neighbor):
                    if far_neighbor in [node, neighbor]:
                        pass
                    else:
                        v2, l2 = _sarc_vector(G, far_neighbor, neighbor)
                        sum_scores = _sarc_score(v1, v2, l1, l2)
                        score = np.max((score, sum_scores))
                score += c_avg_length * avg_length_score
            edges_attr_dict[(node, neighbor)] = score

    edges_attr_dict_keep_max = {}
    for key in edges_attr_dict.keys():
        node_1 = key[0]
        node_2 = key[1]
        max_score = max(
            edges_attr_dict[(node_1, node_2)],
            edges_attr_dict[(node_2, node_1)],
        )
        edges_attr_dict_keep_max[(min(key), max(key))] = max_score
    nx.set_edge_attributes(G, values=edges_attr_dict_keep_max, name="score")


def score_method_1(graph, sg_config, sarc_vector_func, sarc_score_func):
    """# noqa
    Example 'local sarcomere alignment' score (Method 1).
    - Score the edges using the standard SarcGraph pipeline, then scale them.
    """
    # Use sarc_vector_func & sarc_score_func to compute an initial edge score
    score_graph(graph, sg_config, sarc_vector_func, sarc_score_func)

    # Scale as in your snippet: graph[u][v]["score"] /= 3
    for u, v in graph.edges:
        graph[u][v]["score"] = (
            graph[u][v]["score"] / 3
            + (graph.nodes[u]["prob"] + graph.nodes[v]["prob"]) / 2
        ) / 2

    return graph


def score_method_2(
    graph,
    sg_config,
    prune_score_threshold,
    prune_angle_threshold,
    sarcs_angle_func,
):
    """# noqa
    Example 'sarcgraph validity' score (Method 2).
    - Score the edges the same way, then prune, and set final score from 'validity' / 2.
    """
    # 1) Prune edges that fail threshold/angle
    _ = graph_pruning(
        graph,
        score_threshold=prune_score_threshold,
        angle_threshold=prune_angle_threshold,
        sarcs_angle=sarcs_angle_func,
        remove_edges=False,
    )

    # 'graph_pruning' might store 'validity' or any indicator in each edge
    for u, v in graph.edges:
        validity = graph[u][v].get("validity", 1.0)  # fallback if needed
        graph[u][v]["score"] = validity / 2.0


def score_method_3(graph, eta, angle_threshold_deg, iter_num, get_angle_func):
    """
    Example 'global myofibril alignment' score (Method 4).
    - Initially sets 'score' = 1, then uses an iterative update with angles.
    """
    # First assign each edge a unique ID for storing angles
    edges_id = {}
    for i, edge in enumerate(graph.edges):
        edge_key = (min(edge), max(edge))
        edges_id[edge_key] = i
        graph.edges[edge]["score"] = 1.0  # initial

    num_edges = len(graph.edges)
    angles_matrix = np.zeros((num_edges, num_edges))

    # Populate angles_matrix
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                angle = get_angle_func(graph, neighbors[i], node, neighbors[j])
                e1_id = edges_id[
                    (min(neighbors[i], node), max(neighbors[i], node))
                ]
                e2_id = edges_id[
                    (min(node, neighbors[j]), max(node, neighbors[j]))
                ]
                angles_matrix[e1_id, e2_id] = angle
                angles_matrix[e2_id, e1_id] = angle

    # Initialize 'base_score' and node-specific dynamic scores
    for edge in graph.edges:
        graph.edges[edge]["base_score"] = graph.edges[edge]["score"]
        # e.g. 'score_u' = 0, 'score_v' = 0
        u, v = edge
        graph.edges[edge][f"score_{u}"] = 0.0
        graph.edges[edge][f"score_{v}"] = 0.0

    # Iterative updates
    for _ in range(iter_num):
        for u, v in list(graph.edges):
            update_scores(
                graph, u, v, eta, angle_threshold_deg, edges_id, angles_matrix
            )
            update_scores(
                graph, v, u, eta, angle_threshold_deg, edges_id, angles_matrix
            )

    # Combine node-specific scores into a final 'score'
    max_score = 0.0
    for edge in graph.edges:
        u, v = edge
        s = (
            graph.edges[edge][f"score_{u}"] + graph.edges[edge][f"score_{v}"]
        ) / 2.0
        graph.edges[edge]["score"] = s
        if s > max_score:
            max_score = s

    # Normalize by the iteration-based cap: min(iter_num+1, max_score)
    for edge in graph.edges:
        graph.edges[edge]["score"] /= max_score

    return graph


def probabilistic_ensemble(
    base_graph, other_graphs, zdisc_prior=0.5, clip_ranges=None
):
    """# noqa
    Combine edge 'score' attributes from multiple graphs into a single ensemble score.

    This version is flexible:
      - We assume each graph in `other_graphs` is built over the same nodes/edges.
      - For each edge, we retrieve each graph's 'score'.
      - If desired, we clip them to a specified (min,max) range.
      - Then we use a Bernoulli-product trick:
           a = (Π_i s_i) / (zdisc_prior^(N-1))
           b = (Π_i (1 - s_i)) / ((1 - zdisc_prior)^(N-1))
           ensemble_score = a / (a + b)
      - We store the final in base_graph[u][v]['score'].

    You can adapt the exact formula to replicate your snippet
    (e.g. partial combos, or special rules for certain methods).

    Parameters
    ----------
    base_graph : networkx.Graph
        A fresh graph (same node set) where the final ensemble 'score' is stored.
    other_graphs : list of (graph, clip_min, clip_max) or just list of graphs
        The graphs providing edge 'score' to combine. If you want per-graph clipping,
        pass something like [ (g1, 0, 1), (g2, 0.4, 0.6), ... ].
    zdisc_prior : float
        Prior probability for a "true" edge. Used in the Bernoulli mixture formula.
    clip_ranges : list of tuples or None
        If None, no clipping. Otherwise a list of (clip_min, clip_max) for each graph.
        Alternatively you can store each in other_graphs as well.

    Returns
    -------
    base_graph : networkx.Graph
        The same graph but with 'score' replaced by the ensemble combination.
    """
    # If user didn't supply separate clip ranges, assume none
    if clip_ranges is None:
        clip_ranges = [(0.0, 1.0)] * len(other_graphs)

    for u, v in base_graph.edges:
        # Gather scores from each scoring graph
        scores = []
        for idx, G_info in enumerate(other_graphs):
            if isinstance(G_info, tuple):
                # (graph_obj, cmin, cmax)
                G, cmin, cmax = G_info
            else:
                # just the graph object
                G = G_info
                cmin, cmax = clip_ranges[idx]

            s = G[u][v]["score"] if G.has_edge(u, v) else 0.0
            s = np.clip(s, cmin, cmax)
            scores.append(s)

        # Compute product of s_i and product of (1 - s_i) By default, each
        # edge's prior is zdisc_prior, so we do (N-1) in exponent if you want
        # that formula or adapt to your snippet's partial combos, etc.
        if len(scores) == 0:
            base_graph[u][v]["score"] = 0.0
            continue

        prod_s = 1.0
        prod_1_minus_s = 1.0
        for s_i in scores:
            prod_s *= s_i
            prod_1_minus_s *= 1 - s_i

        # Example formula: a = prod_s / (zdisc_prior^(N-1)), b = ...
        # but you can adapt the exponent. If you have N methods, you might
        # want zdisc_prior^N, etc.
        N = len(scores)
        a = prod_s / (zdisc_prior ** (N - 1)) if N > 1 else prod_s
        b = (
            prod_1_minus_s / ((1 - zdisc_prior) ** (N - 1))
            if N > 1
            else prod_1_minus_s
        )

        final_score = a / (a + b) if (a + b) > 0 else 0.0
        base_graph[u][v]["score"] = final_score

    return base_graph


def build_and_score_all(zdiscs_corrected, new_probs, sg, configs):
    """
      - Create graphs 1-4 with different scoring
      - Then ensemble them in a final graph
      - Return the resulting list or single ensemble graph

    This is just a reference outline. Adjust to your data and config usage.
    """
    # 1) Graph #1: local sarcomere alignment & z-disc probability (averaged)
    graph_1 = create_knn_graph(zdiscs_corrected, new_probs, k=5)
    score_method_1(graph_1, sg.config, sg._sarc_vector, sg._sarc_score)

    # 2) Graph #2: validity score, then prune
    graph_2 = copy.deepcopy(graph_1)
    score_method_2(
        graph_2,
        sg.config,
        configs.score_threshold,
        configs.angle_threshold,
        sg._sarcs_angle,
    )

    # 3) Graph #3: global alignment
    graph_3 = create_knn_graph(zdiscs_corrected, new_probs, k=5)
    score_method_3(
        graph_3,
        eta=0.8,
        angle_threshold_deg=30,  # for example
        iter_num=6,
        get_angle_func=get_angle,
    )

    # 4) Combine them. A new base graph:
    graph_ensemble = create_knn_graph(zdiscs_corrected, new_probs, k=5)

    # Possibly define clip ranges if you want to clamp method #2 scores to
    # [0.4, 0.6], etc. E.g. (graph, min, max)
    scoring_graphs = [
        (graph_1, 0.0, 1.0),  # Method #1
        (graph_2, 0.4, 0.6),  # Method #2 (clamp to [0.4, 0.6])
        (graph_3, 0.0, 1.0),  # Method #3
    ]

    probabilistic_ensemble(
        base_graph=graph_ensemble, other_graphs=scoring_graphs, zdisc_prior=0.5
    )

    return graph_1, graph_2, graph_3, graph_ensemble
