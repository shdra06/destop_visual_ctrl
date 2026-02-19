import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalizes landmarks to be relative to the wrist and scale-invariant.
    Args:
        landmarks: List of objects with .x, .y, .z attributes OR list of [x, y, z] lists.
    Returns:
        List of 63 floats (normalized flat coordinates).
    """
    # Handle different input formats (MediaPipe objects vs lists)
    if hasattr(landmarks[0], 'x'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        coords = np.array(landmarks)
        
    # Relative usage: subtract wrist (landmark 0)
    base_x, base_y, base_z = coords[0]
    coords[:, 0] -= base_x
    coords[:, 1] -= base_y
    coords[:, 2] -= base_z
    
    # Scale invariance: divide by max absolute value
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
        
    return coords.flatten().tolist()
