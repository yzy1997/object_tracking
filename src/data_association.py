import numpy as np
from scipy.optimize import linear_sum_assignment

def associate_detections_to_tracks(detections, predictions, max_distance=20):
    """
    detections: [(x,y), ...]
    predictions: [(x,y), ...]
    max_distance: 超过这个距离视为不匹配

    Returns:
        matches: [(det_idx, track_idx), ...]
        unmatched_detections: [det_idx, ...]
        unmatched_tracks: [track_idx, ...]
    """
    if len(predictions) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(predictions)))

    cost_matrix = np.zeros((len(detections), len(predictions)), dtype=np.float32)
    for i, det in enumerate(detections):
        for j, pred in enumerate(predictions):
            cost_matrix[i, j] = np.linalg.norm(np.array(det) - np.array(pred))

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(predictions)))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= max_distance:
            matches.append((r, c))
            unmatched_detections.remove(r)
            unmatched_tracks.remove(c)

    return matches, unmatched_detections, unmatched_tracks
