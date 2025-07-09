import copy

import numpy as np
import pandas as pd
import networkx as nx

# Import the custom modules we created
from sarcomere_detection.preprocessing import (
    pad_image,
    filter_contours,
    correct_zdiscs,
)
from sarcomere_detection.graph_operations import (
    create_knn_graph,
    graph_pruning,
    myofibril_extension,
)
from sarcomere_detection.graph_scoring import build_and_score_all
from sarcomere_detection.feature_extraction import (
    compute_z_disc_probability_weighted_area,
    extract_myofibril_features,
)

# Constants
PAD = 64
PROB_THRESH = 0.2
INTENSITY_THRESH = 0.6
INITIAL_POINTS = 15


def process_single_sample(
    sample_id: str,
    raw: np.ndarray,
    cell_mask: np.ndarray,
    probabilities: np.ndarray,
    sg,
    PIXEL_TO_MICRON: float,
):
    """
    Run the SarcGraph pipeline on provided sample inputs.

    Args:
        sample_id: Identifier for this sample
        raw: Raw cell image (2D array)
        contours: List of contour arrays
        probabilities: Array of contour probabilities
        cell_mask: Binary mask of the cell region
        output_dir: Directory to save outputs

    Returns:
        A dict containing:
            - features: Extracted feature dict
            - graph: networkx Graph of final myofibril structure
            - graph_path: File path where the graph pickle was saved
            - features_df: Pandas DataFrame of features (if conversion used)
    """
    configs = sg.config

    # 1) Pad raw image
    raw_padded = pad_image(raw, pad=PAD)
    _ = sg.zdisc_segmentation(raw_frames=raw)
    contours = np.load(
        configs.output_dir + "/contours.npy", allow_pickle=True
    )[0]

    # 2) Filter contours by probability
    filtered_contours, filtered_probs = filter_contours(
        contours, probabilities, PROB_THRESH
    )

    # 3) Correct Z-disc locations
    zdiscs_corrected, new_probs = correct_zdiscs(
        raw_padded=raw_padded,
        contours=filtered_contours,
        probs=filtered_probs,
        intensity_thresh=INTENSITY_THRESH,
        initial_points=INITIAL_POINTS,
        pad=PAD,
    )

    # 4) Build and score ensemble graphs
    _, _, _, graph_ensemble = build_and_score_all(
        zdiscs_corrected, new_probs, sg, configs
    )

    # 5) Create k-NN graph and copy ensemble scores
    graph = create_knn_graph(
        zdiscs_corrected, new_probs, k=sg.config.num_neighbors
    )
    for u, v in graph.edges():
        graph.edges[u, v]["score"] = (
            graph_ensemble.edges[u, v]["score"]
            if graph_ensemble.has_edge(u, v)
            else 0.0
        )

    # 6) Prune and extend
    edges_to_remove = graph_pruning(
        graph_ensemble,
        score_threshold=0.8,
        angle_threshold=sg.config.angle_threshold,
        sarcs_angle=sg._sarcs_angle,
        remove_edges=False,
    )
    graph.remove_edges_from(edges_to_remove)
    graph = myofibril_extension(graph, zdiscs_corrected, new_probs)

    # Remove trivial 2-node components
    final_graph = copy.deepcopy(graph)
    for comp in list(nx.connected_components(final_graph)):
        if len(comp) == 2:
            final_graph.remove_edge(*comp)

    # 7) Extract features
    z_disc_prob_area = compute_z_disc_probability_weighted_area(
        contours, probabilities
    )
    features = extract_myofibril_features(
        final_graph,
        PIXEL_TO_MICRON,
        sample_id=sample_id,
        cell_mask=cell_mask,
        z_disc_probability_weighted_area=z_disc_prob_area,
        z_disc_probabilities=probabilities,
    )

    # Optionally convert features to DataFrame
    features_df = (
        pd.DataFrame([features]) if isinstance(features, dict) else features
    )

    return features_df, final_graph
