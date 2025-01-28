import numpy as np
import networkx as nx
import cv2


def compute_z_disc_probability_weighted_area(contours, probabilities):
    """
    Compute the total area of all contours weighted by their probabilities.

    Parameters
    ----------
    contours : np.ndarray (object array)
        An array of contour points, each an Nx2 float array representing (x,y).
    probabilities : np.ndarray
        A 1D array of probabilities for each contour, same length as 'contours'.

    Returns
    -------
    float
        Sum of (contour_area * contour_probability) over all contours.
        If 'contours' or 'probabilities' is empty, returns 0.
    """
    if len(contours) == 0 or len(probabilities) == 0:
        return 0.0
    
    # Calculate contour areas
    contours_area = np.array([
        cv2.contourArea(c.astype(np.int32)) for c in contours
    ])

    # Multiply each contour's area by its probability and sum
    weighted_area = np.sum(probabilities * contours_area)
    return weighted_area


def orientation_tensor(angles):
    """
    Compute the orientation tensor from an array of angles.

    Parameters
    ----------
    angles : array-like
        Array of angles in radians.

    Returns
    -------
    ot : np.ndarray, shape (2,2)
        The 2x2 orientation tensor averaged over all angles.
    """
    rxx = np.cos(angles) ** 2
    rxy = np.cos(angles) * np.sin(angles)
    ryy = np.sin(angles) ** 2

    # n is shape (2,2,len(angles)): n[0,0,:] = rxx, etc.
    n = np.array([[rxx, rxy],
                  [rxy, ryy]])

    # t is shape (2,2,len(angles)) => 2*n - I
    t = 2.0 * n - np.eye(2).reshape(2, 2, 1)

    # Average across the angle dimension
    ot = np.mean(t, axis=2)
    return ot


def extract_myofibril_features(
    graph,
    PIXEL_TO_MICRON,
    sample_id=None,
    cell_mask=None,
    z_disc_probability_weighted_area=None,
    z_disc_probabilities=None,
):
    """
    Extract sarcomere and myofibril features from a (possibly pruned & extended) graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose edges correspond to sarcomeres. Each node has:
            - 'pos': np.array([x, y]) coordinate
    sample_id : int, optional
        Identifier for this sample (if you want to store it in the output).
    cell_mask : np.ndarray, optional
        A 2D boolean array specifying the cell area. Used to compute 'cell_area'.
        If None, 'cell_area' is set to np.nan.
    z_disc_probability_weighted_area : float, optional
        Sum of (area * probability) for all Z-discs, if precomputed.
        If None, 'z_disc_probability_weighted_area' is set to np.nan.
    z_disc_probabilities : float, optional
        Probability of each contour being a Z-disc.

    Returns
    -------
    features : dict
        A dictionary with keys:
            'sample_id' : int or None
            'sarcomeres_mean_length' : float
            'sarcomeres_std_length' : float
            'sarcomeres_count' : int
            'myofibrils_count' : int
            'sarcomeres_per_myofibril': float
            'oop' : float
            'cell_area' : float
            'z_disc_probability_weighted_area' : float
            'z_disc_ratio' : float
    """
    # Compute cell_area if mask is given
    if cell_mask is not None:
        cell_area = np.sum(cell_mask)
    else:
        cell_area = np.nan

    # Default to np.nan if user doesn't provide weighted area
    if z_disc_probability_weighted_area is None:
        z_dwa = np.nan
    else:
        z_dwa = z_disc_probability_weighted_area

    # Compute z_disc_ratio
    if z_disc_probabilities is not None and len(z_disc_probabilities) > 0:
        z_disc_ratio = np.sum(z_disc_probabilities > 0.5) / len(z_disc_probabilities)
    else:
        z_disc_ratio = np.nan
        
    lengths = []
    angles = []
    # 1. Gather edge lengths and orientation angles
    for (u, v) in graph.edges:
        pos_u = graph.nodes[u]['pos']
        pos_v = graph.nodes[v]['pos']

        # Sarcomere length
        length = np.linalg.norm(pos_v - pos_u)
        lengths.append(length)

        # Orientation angle
        vec = pos_v - pos_u
        # Flip sign if the x-component is negative (to keep angles consistent)
        if vec[0] < 0:
            vec = -vec

        angles.append(np.pi + np.arctan2(-vec[0], vec[1]))

    if len(angles) == 0:
        # No edges => no sarcomeres => return mostly np.nan
        return {
            "sample_id": sample_id,
            "sarcomeres_mean_length": np.nan,
            "sarcomeres_std_length": np.nan,
            "sarcomeres_count": 0,
            "myofibrils_count": 0,
            "sarcomeres_per_myofibril": np.nan,
            "oop": np.nan,
            "cell_area": cell_area * PIXEL_TO_MICRON**2,
            "z_disc_probability_weighted_area": z_dwa * PIXEL_TO_MICRON**2,
            "z_disc_ratio": z_disc_ratio,
        }

    # 2. Compute orientation tensor & extract OOP
    ot = orientation_tensor(np.array(angles))
    eigvals, _ = np.linalg.eig(ot)
    oop = float(np.max(eigvals))  # orientation order parameter

    # 3. Count sarcomeres & myofibrils
    sarcs_counter = 0
    myo_counter = 0
    for component in nx.connected_components(graph):
        c_size = len(component)
        # If c_size > 2, we consider it a legitimate myofibril
        if c_size > 2:
            myo_counter += 1
            # A chain of N nodes has (N - 1) sarcomeres
            sarcs_counter += (c_size - 1)

    if myo_counter == 0:
        sarcomeres_per_myofibril = np.nan
    else:
        sarcomeres_per_myofibril = float(sarcs_counter) / float(myo_counter)

    # 4. Assemble final feature dictionary
    features = {
        "sample_id": sample_id,
        "sarcomeres_mean_length": float(np.mean(lengths)) * PIXEL_TO_MICRON,
        "sarcomeres_std_length": float(np.std(lengths)) * PIXEL_TO_MICRON,
        "sarcomeres_count": sarcs_counter,
        "myofibrils_count": myo_counter,
        "sarcomeres_per_myofibril": sarcomeres_per_myofibril,
        "oop": oop,
        "cell_area": float(cell_area) * PIXEL_TO_MICRON**2,
        "z_disc_probability_weighted_area": float(z_dwa) * PIXEL_TO_MICRON**2,
        "z_disc_ratio": z_disc_ratio,
    }
    
    return features