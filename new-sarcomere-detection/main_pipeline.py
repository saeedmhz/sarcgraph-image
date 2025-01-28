# main_pipeline.py

import os
import copy
import pickle

import numpy as np
import pandas as pd
import networkx as nx

# Import the custom modules we created
from preprocessing import load_sample_data, pad_image, filter_contours, correct_zdiscs
from graph_operations import  create_knn_graph, graph_pruning, myofibril_extension
from graph_scoring import build_and_score_all
from feature_extraction import compute_z_disc_probability_weighted_area, extract_myofibril_features


from sarcgraph import SarcGraph

sg = SarcGraph(
    num_neighbors=5,
    angle_threshold=1.75,
    max_sarc_length=20.0,
    zdisc_min_length=15,
    zdisc_max_length=200,
    input_type="image",
    sigma=2.0
)

configs = sg.config


# Global or local constants
PAD = 64
PROB_THRESH = 0.2
INTENSITY_THRESH = 0.6
INITIAL_POINTS = 15

ETA = 0.8
ANG_THRESHOLD = 30.0
ITER_NUM = 6

PRIOR = 0.5

PIXEL_TO_MICRON = 207.5 / 1736

def process_single_sample(sample_id, file_addresses, output_dir="sarcgraph-extracted-features"):
    """
    Orchestrate the entire pipeline for one sample_id:
     1) Load & pad image
     2) Filter contours
     3) Correct Z-disc locations
     4) Build & score multiple graphs
     5) Ensemble
     6) Prune, Myofibril extend
     7) Extract features
     8) Save final graph & return feature dict
    """
    # Step 1: Load data
    raw, probabilities, zdiscs_df, contours, cell_mask = load_sample_data(
        sample_id,
        file_addresses['raw_dir'],
        file_addresses['dino_dir'],
        file_addresses['simclr_dir'],
        file_addresses['zdiscs_dir'],
        file_addresses['mask_dir'],
    ) # raw_dir, dino_dir, simclr_dir, zdiscs_dir, mask_dir
    
    # Step 2: Pad raw image
    raw_padded = pad_image(raw, pad=PAD)

    # Step 3: Filter contours by probability
    filtered_contours, filtered_probs = filter_contours(contours, probabilities, PROB_THRESH)

    # Step 4: Correct Z-disc locations
    zdiscs_corrected, new_probs = correct_zdiscs(
        raw_padded=raw_padded,
        contours=filtered_contours,
        probs=filtered_probs,
        intensity_thresh=INTENSITY_THRESH,
        initial_points=INITIAL_POINTS,
        pad=PAD
    )

    # Step 5: Build multiple scoring graphs (Method 1..4)
    _, _, _, graph_ensemble = build_and_score_all(zdiscs_corrected, new_probs, sg, configs)


    # Step 6: Pruning & Myofibril extension
    graph = create_knn_graph(zdiscs_corrected, new_probs, k=5)
    
    # Copy ensemble scores
    for edge in graph.edges:
        if graph_ensemble.has_edge(*edge):
            graph.edges[edge]['score'] = graph_ensemble.edges[edge]['score']
        else:
            graph.edges[edge]['score'] = 0.0

    edges_to_remove = graph_pruning(
        graph_ensemble,
        score_threshold=0.8,
        angle_threshold=1.7,
        sarcs_angle=sg._sarcs_angle,
        remove_edges=False
    )
    graph.remove_edges_from(edges_to_remove)

    # Myofibril extension
    graph = myofibril_extension(graph, zdiscs_corrected, new_probs)

    # Clean out 2-node components
    final_graph = copy.deepcopy(graph)
    for myo in nx.connected_components(final_graph):
        if len(myo) == 2:
            final_graph.remove_edge(*list(myo))

    # Step 7: Extract features
    z_disc_prob_area = compute_z_disc_probability_weighted_area(contours, probabilities)

    features = extract_myofibril_features(
        final_graph,
        PIXEL_TO_MICRON,
        sample_id=sample_id,
        cell_mask=cell_mask,
        z_disc_probability_weighted_area=z_disc_prob_area,
        z_disc_probabilities=probabilities
    )

    # Optionally, save the final graph to pickle
    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, f"graph-{sample_id}.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(final_graph, f)

    return features