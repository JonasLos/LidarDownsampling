def distance_adaptive_downsample(points: np.ndarray, near_keep_prob=0.1, far_keep_prob=1.0, max_distance=100.0) -> np.ndarray:
    """
    Downsample a point cloud based on distance from origin. Retains more distant points.

    Args:
        points (np.ndarray): Nx3 or Nx4 array of points (x, y, z, [intensity]).
        near_keep_prob (float): Probability of keeping points very close to sensor (e.g., < 1m).
        far_keep_prob (float): Probability of keeping farthest points (default 1.0).
        max_distance (float): Distance at which far_keep_prob applies.

    Returns:
        np.ndarray: Downsampled point cloud.
    """
    coords = points[:, :3]
    distances = np.linalg.norm(coords, axis=1)

    # Probability Calculation
    probs = near_keep_prob + (far_keep_prob - near_keep_prob) * (distances / max_distance)
    probs = np.clip(probs, near_keep_prob, far_keep_prob)

    keep_mask = np.random.rand(points.shape[0]) < probs
    return points[keep_mask]
