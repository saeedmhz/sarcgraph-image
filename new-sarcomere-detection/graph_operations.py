import copy
import networkx as nx
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial import distance_matrix

def create_knn_graph(points, probs, k, lower_threshold=None, upper_threshold=None):
    """
    Build a k-NN graph using mutual neighbors, optionally filtering edges by distance.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with x,y coordinates of points (z-discs).
    probs : np.ndarray
        Array of length N with probability/confidence scores for each point.
    k : int
        Number of neighbors to connect for each point.
    lower_threshold : float, optional
        If provided, edges are only kept if their length is > lower_threshold.
    upper_threshold : float, optional
        If provided, edges are only kept if their length is < upper_threshold.

    Returns
    -------
    G : networkx.Graph
        The resulting k-NN graph, where each node has attributes: "pos" and "prob".
    """
    # Calculate the distance matrix
    dists = distance_matrix(points, points)

    # For each point, find the indices of the k nearest neighbors (excluding itself)
    nearest_neighbors = np.argsort(dists, axis=1)[:, 1 : k + 1]  # slice out the 0th, which is the point itself

    # Create the graph
    G = nx.Graph()

    # Add nodes
    for idx, point in enumerate(points):
        G.add_node(idx, pos=point, prob=probs[idx])

    # Add edges for mutual nearest neighbors within optional distance thresholds
    num_points = len(points)
    for i in range(num_points):
        for j in nearest_neighbors[i]:
            # Check if i is also in j's neighbor list (mutual k-NN)
            if i in nearest_neighbors[j]:
                dist_ij = dists[i, j]
                # Apply lower/upper distance thresholds if provided
                if ((lower_threshold is None or dist_ij > lower_threshold) and
                    (upper_threshold is None or dist_ij < upper_threshold)):
                    G.add_edge(i, j)

    return G


def get_angle(graph, n1, n2, n3):
    """
    Calculate the angle (in degrees) formed at node n2 by edges (n1->n2) and (n2->n3).

    Parameters
    ----------
    graph : networkx.Graph
        A graph whose nodes have a "pos" attribute (x,y coordinates).
    n1, n2, n3 : int
        Node indices in the graph.

    Returns
    -------
    float
        Angle in degrees, from 0 to 180.
    """
    p1 = graph.nodes[n1]['pos']
    p2 = graph.nodes[n2]['pos']
    p3 = graph.nodes[n3]['pos']

    v12 = p2 - p1
    v23 = p3 - p2

    l12 = np.linalg.norm(v12)
    l23 = np.linalg.norm(v23)

    # Avoid divide-by-zero
    if l12 == 0 or l23 == 0:
        return 0.0

    dot_prod = np.dot(v12, v23) / (l12 * l23)
    dot_prod = np.clip(dot_prod, -1, 1)  # clip to avoid floating precision issues
    angle_rad = np.arccos(dot_prod)

    return 180.0 * angle_rad / np.pi


def update_scores(graph, u, v, eta, threshold, edges_id, angles_matrix):
    """
    Update the dynamic edge score for edge (u, v) based on angles with neighbors of v.

    This function references an external 'edges_id' dictionary mapping edges to indices,
    and an 'angles_matrix' storing angles between edges. Each edge in the graph is 
    expected to have a 'base_score' plus dynamic scores keyed by 'score_{node}'.

    This function implements the Global Myofibril Alignment (GMA) algorithm from the paper.

    Parameters
    ----------
    graph : networkx.Graph
        The graph whose edges are being updated in-place.
    u, v : int
        Node indices for the edge.
    eta : float
        Weighting factor for the final score.
    threshold : float
        Angular threshold in degrees. Only neighbors whose angles are below this
        will contribute to the updated score.
    edges_id : dict
        Dictionary that maps the tuple (min_node, max_node) -> integer index for angles_matrix.
    angles_matrix : np.ndarray
        2D array of angles [edge_index1, edge_index2] -> angle in degrees.

    Returns
    -------
    None
        (Modifies graph[u][v] in place, adding or updating the key f'score_{v}'.)
    """
    # Retrieve the unique ID for edge (u,v)
    edge_key = (min(u, v), max(u, v))
    edge_idx = edges_id[edge_key]

    neighbors_of_v = list(graph.neighbors(v))

    angles = []
    scores = []
    for neighbor in neighbors_of_v:
        if neighbor == u:
            continue

        other_key = (min(neighbor, v), max(neighbor, v))
        other_idx = edges_id[other_key]

        angle_deg = angles_matrix[edge_idx, other_idx]
        if angle_deg < threshold:
            angles.append(angle_deg)
            base_score = graph[v][neighbor].get('base_score', 0.0)
            neighbor_score_key = f'score_{neighbor}'
            dynamic_score = graph[v][neighbor].get(neighbor_score_key, 0.0)
            scores.append(base_score + dynamic_score)

    if angles:
        angles = np.array(angles)
        scores = np.array(scores)
        # Weighted by cos(angle) (converted to radians)
        cos_factors = np.cos(np.pi * angles / 180.0)
        combined = cos_factors * scores

        max_val = np.max(combined)
        graph[u][v][f'score_{v}'] = eta * max_val


def graph_pruning(
    graph,
    score_threshold,
    angle_threshold,
    sarcs_angle,
    remove_edges=True
):
    """
    Prune edges in a graph based on 'score' and angle constraints.

    Steps:
      1. Initialize each edge's "validity" to 0.
      2. For each node:
         - Gather all neighbor edges with their vectors & scores.
         - If the highest score among these neighbors is above score_threshold:
             * Mark that "best" edge's validity += 1.
             * Then scan remaining edges in descending order of score:
               - If the score is also above score_threshold
               - Compute the angle between this vector and the "best" vector
               - If that angle is > angle_threshold, mark its validity += 1 and break.
      3. Finally, any edge whose total "validity" is < 2 is removed or returned.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with edges that have a 'score' attribute.
    score_threshold : float
        Minimum score for edges to be considered in the pruning logic.
    angle_threshold : float
        If the angle between two vectors is greater than this threshold,
        the second vector is also marked valid.
    sarcs_angle : callable
        A function that calculates the angle between two vectors given
        (vector1, vector2, length1, length2).
    remove_edges : bool, optional
        If True, remove edges with validity < 2 from the graph in-place.
        If False, return them in a list.

    Returns
    -------
    None or list of edges
        If remove_edges=True, returns None after removing failing edges.
        If remove_edges=False, returns the list of edges to be removed.
    """
    # Set 'validity' = 0 for all edges
    nx.set_edge_attributes(graph, values=0, name="validity")

    # For each node, process neighbor edges
    for node in range(graph.number_of_nodes()):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue

        vectors = []
        scores = []
        for nbr in neighbors:
            vec = graph.nodes[nbr]["pos"] - graph.nodes[node]["pos"]
            vectors.append(vec)
            scores.append(graph[node][nbr]["score"])

        # If there's at least one neighbor with score > threshold
        if not scores or (np.max(scores) <= score_threshold):
            continue

        # Sort neighbors by descending score
        sort_indices = np.argsort(scores)[::-1]
        best_idx = sort_indices[0]
        best_vector = vectors[best_idx]
        # Mark the best edge valid
        graph[node][neighbors[best_idx]]["validity"] += 1

        # Check other edges for angle compatibility
        for idx in sort_indices[1:]:
            s = scores[idx]
            if s <= score_threshold:
                continue

            v = vectors[idx]
            l1 = np.linalg.norm(best_vector)
            l2 = np.linalg.norm(v)
            theta = sarcs_angle(v, best_vector, l1, l2)

            # If angle is bigger than angle_threshold, mark validity and stop
            if theta > angle_threshold:
                graph[node][neighbors[idx]]["validity"] += 1
                break

    # After all nodes are processed, collect edges whose validity < 2
    edges2remove = []
    for edge in graph.edges():
        if graph.edges[edge]["validity"] < 2:
            edges2remove.append(edge)

    if remove_edges:
        graph.remove_edges_from(edges2remove)
        return None
    else:
        return edges2remove


def myofibril_extension(
    graph,
    zdiscs,
    probs,
    prob_thresh=0.5
):
    """
    Extend existing myofibrils by searching for plausible new z-discs.

    The logic cycles multiple times (fixed 5 iterations), checking endpoints
    (nodes with degree==1) in each connected component. It predicts a new z-disc
    location, queries a KDTree for the nearest actual z-disc, and evaluates 
    whether to merge or create new nodes/edges under certain length and angular 
    constraints. The final result is returned as a copied graph with small 
    two-node components removed.

    Parameters
    ----------
    graph : networkx.Graph
        The starting graph representing partial myofibrils. Nodes must have a 'pos' attribute
        with (x,y) location.
    zdiscs : np.ndarray
        Array of shape (N, 2) containing the possible z-disc coordinates.
    probs : np.ndarray
        Probabilities associated with each z-disc in the same order as `zdiscs`.
    prob_thresh : float, optional
        Minimum probability required for a z-disc to be considered valid
        for extension.

    Returns
    -------
    graph_copy : networkx.Graph
        A new graph where edges have been extended from existing myofibrils if conditions
        are met. Small two-node components are removed.
    """
    # Repeat the extension process for a fixed number of iterations
    for _ in range(5):
        # Current node degrees
        nodes_degree = dict(graph.degree())

        # Build KDTree for quick nearest-neighbor queries
        kd_tree = KDTree(zdiscs)

        # Hard-coded thresholds (kept identical to original logic)
        min_length = 10
        max_length = 20
        angle_threshold = 180 - 1.75 * 90  # if referencing sg.config._angle_threshold = 1.75
        min_distance = 7  # e.g. "np.mean(dists) - 2 * np.std(dists)" used previously

        # Identify connected components (myofibrils)
        myofibrils = list(nx.connected_components(graph))
        for myo in myofibrils:
            if len(myo) < 2:
                continue

            # Find "endpoints" (degree == 1)
            end_nodes = [node for node in myo if nodes_degree[node] == 1]

            # Extend each end
            for end_node in end_nodes:
                end_neighbors = list(graph.neighbors(end_node))
                if not end_neighbors:
                    continue

                end_node_neighbor = end_neighbors[0]
                end_sarc = (graph.nodes[end_node]['pos']
                            - graph.nodes[end_node_neighbor]['pos'])

                # Predict a new z-disc position
                pred_zdisc = graph.nodes[end_node]['pos'] + end_sarc

                # Find closest real z-disc
                distance, target_node = kd_tree.query(pred_zdisc)
                if distance > min_distance:
                    continue

                # Average predicted location with the real z-disc location
                pred_zdisc = (pred_zdisc + graph.nodes[target_node]['pos']) / 2
                pred_sarc = pred_zdisc - graph.nodes[end_node]['pos']
                pred_sarc_length = np.linalg.norm(pred_sarc)

                # Check angle
                sarc_cosine = np.dot(pred_sarc, end_sarc) / (
                    np.linalg.norm(pred_sarc) * np.linalg.norm(end_sarc) + 1e-8
                )
                pred_sarc_angle_end = 180.0 * np.arccos(np.clip(sarc_cosine, -1, 1)) / np.pi

                # Filter by length, angle, probability
                if (pred_sarc_length < min_length or
                    pred_sarc_length > max_length or
                    pred_sarc_angle_end > angle_threshold or
                    probs[target_node] < prob_thresh):
                    continue

                # Case 1: if target_node is unassigned (degree 0)
                if nodes_degree[target_node] == 0:
                    graph.nodes[target_node]['pos'] = pred_zdisc
                    graph.add_edge(end_node, target_node)
                    nodes_degree[end_node] += 1
                    nodes_degree[target_node] += 1
                    continue

                # Case 2: if target_node is an endpoint as well (degree 1)
                if nodes_degree[target_node] == 1:
                    tgt_neighbor = list(graph.neighbors(target_node))[0]
                    target_sarc = (graph.nodes[tgt_neighbor]['pos']
                                   - graph.nodes[target_node]['pos'])
                    angle_tgt = 180.0 * np.arccos(
                        np.dot(pred_sarc, target_sarc) / (
                            np.linalg.norm(pred_sarc) * np.linalg.norm(target_sarc) + 1e-8
                        )
                    ) / np.pi
                    if angle_tgt < angle_threshold:
                        graph.nodes[target_node]['pos'] = pred_zdisc
                        graph.add_edge(end_node, target_node)
                        nodes_degree[end_node] += 1
                        nodes_degree[target_node] += 1
                        continue

                # Further extension attempt (second extension)
                pred_zdisc_2 = pred_zdisc + (pred_sarc + end_sarc) / 2
                distance, target_node_2 = kd_tree.query(pred_zdisc_2)
                if distance > min_distance:
                    continue

                pred_zdisc_2 = (pred_zdisc_2 + graph.nodes[target_node_2]['pos']) / 2
                pred_sarc_2 = pred_zdisc_2 - pred_zdisc
                pred_sarc_2_length = np.linalg.norm(pred_sarc_2)
                angle_sarc_2 = 180.0 * np.arccos(
                    np.dot(pred_sarc_2, pred_sarc) / (
                        np.linalg.norm(pred_sarc) * np.linalg.norm(pred_sarc_2) + 1e-8
                    )
                ) / np.pi

                if (pred_sarc_2_length < min_length or
                    pred_sarc_2_length > max_length or
                    angle_sarc_2 > angle_threshold or
                    probs[target_node_2] < prob_thresh):
                    continue

                # If the second target node is unassigned
                if nodes_degree[target_node_2] == 0:
                    new_idx = graph.number_of_nodes()
                    graph.add_node(new_idx)
                    graph.nodes[new_idx]['pos'] = pred_zdisc
                    graph.nodes[new_idx]['prob'] = graph.nodes[target_node]['prob']
                    graph.add_edge(end_node, new_idx)
                    nodes_degree[end_node] += 1

                    graph.nodes[target_node_2]['pos'] = pred_zdisc_2
                    graph.add_edge(target_node_2, new_idx)
                    nodes_degree[end_node] += 1
                    continue

                # Or if the second target node is an endpoint
                if nodes_degree[target_node_2] == 1:
                    tgt_2_neighbor = list(graph.neighbors(target_node_2))[0]
                    target_sarc_2 = (graph.nodes[tgt_2_neighbor]['pos']
                                     - graph.nodes[target_node_2]['pos'])
                    angle_tgt_2 = 180.0 * np.arccos(
                        np.dot(pred_sarc_2, target_sarc_2) / (
                            np.linalg.norm(pred_sarc_2) * np.linalg.norm(target_sarc_2) + 1e-8
                        )
                    ) / np.pi

                    if angle_tgt_2 < angle_threshold:
                        new_idx = graph.number_of_nodes()
                        graph.add_node(new_idx)
                        graph.nodes[new_idx]['pos'] = pred_zdisc
                        graph.nodes[new_idx]['prob'] = graph.nodes[target_node]['prob']
                        graph.add_edge(end_node, new_idx)
                        nodes_degree[end_node] += 1

                        graph.nodes[target_node_2]['pos'] = pred_zdisc_2
                        graph.add_edge(target_node_2, new_idx)
                        nodes_degree[end_node] += 1
                        continue

    # After extension attempts, copy & remove small 2-node components
    graph_copy = copy.deepcopy(graph)
    for myo in nx.connected_components(graph_copy):
        if len(myo) == 2:
            graph_copy.remove_edge(*list(myo))

    return graph_copy
