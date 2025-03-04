import numpy as np

def integrate_events_to_one_frame_1bit_optimized_numpy(events: np.ndarray):
    """
    Integrate all DVS camera events into a single frame.

    Parameters:
        events (np.ndarray): Array containing event data. shape: [[x, y, polarity], ...]

    Returns:
        frame (np.ndarray): Integrated frame as a numpy array with shape [2, 86, 65]. (maxpool2d, stride=4)
    """
    if len(events) == 0:
        return np.zeros((2, 86, 65), dtype=np.uint8)

    # Initialize an empty frame with shape [1, 2, x_max, y_max]
    frame = np.zeros((2, 86, 65), dtype=np.uint8)

    # Extract the x, y, and polarity columns as NumPy arrays
    x = events[['x']]
    y = events[['y']]
    polarity = events[['polarity']]

    x = np.array(x, dtype=np.int16)
    y = np.array(y, dtype=np.int16)
    polarity = np.array(polarity, dtype=np.int8)

    # maxpool2d, stride=4
    x = x // 4
    x = np.where(x > 85, 85, x)
    y = y // 4
    y = np.where(y > 64, 64, y)

    # Use NumPy's advanced indexing to set the frame values
    frame[polarity, x, y] = 1

    return frame