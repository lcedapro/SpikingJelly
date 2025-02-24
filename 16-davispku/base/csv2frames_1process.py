import numpy as np
import pandas as pd

def process_event_batch(batch, min_timestamp, duration, x_max, y_max):
    """
    Process a batch of events and integrate them into frames.

    Parameters:
        batch (pd.DataFrame): Batch of events.
        min_timestamp (int): Minimum timestamp from the dataset.
        duration (int): Duration for each frame integration in microseconds.
        x_max (int): Maximum x coordinate.
        y_max (int): Maximum y coordinate.

    Returns:
        np.ndarray: Partial frames integrated from the batch.
    """
    num_frames = (batch['timestamp'].iloc[-1] - min_timestamp) // duration + 1
    partial_frames = np.zeros((num_frames, 2, x_max, y_max), dtype=np.uint8)

    for _, event in batch.iterrows():
        timestamp = event['timestamp']
        x = event['x']
        y = event['y']
        polarity = event['polarity']
        frame_idx = (timestamp - min_timestamp) // duration
        partial_frames[frame_idx, polarity, x, y] += 1

    return partial_frames

def integrate_events_to_frames(df: pd.DataFrame, duration: int, max_frames: int = None):
    """
    Integrate DVS camera events into frames.

    Parameters:
        df (pd.DataFrame): DataFrame containing event data.
        duration (int): Duration for each frame integration in microseconds.
        max_frames (int, optional): Maximum number of frames to output. Defaults to None (no limit).

    Returns:
        np.ndarray: Integrated frames as a numpy array with shape [N, 2, 346, 260].
    """
    # Constants for DAVIS346 sensor resolution
    x_max, y_max = 346, 260

    # Get the minimum and maximum timestamps
    min_timestamp = df['timestamp'].iloc[0]
    max_timestamp = df['timestamp'].iloc[-1]

    # Calculate the number of frames
    total_num_frames = (max_timestamp - min_timestamp) // duration + 1

    # Apply max_frames limit
    if max_frames is not None:
        total_num_frames = min(total_num_frames, max_frames)

    # Assign each event to a frame index
    df['frame_idx'] = (df['timestamp'] - min_timestamp) // duration

    # Filter events to only include those within max_frames
    if max_frames is not None:
        max_timestamp = min_timestamp + max_frames * duration
        df = df[df['timestamp'] < max_timestamp]

    # Group by frame index to ensure no cross-frame split
    grouped_batches = [group for _, group in df.groupby('frame_idx')]

    # Process each batch sequentially (single-threaded)
    frames = np.zeros((total_num_frames, 2, x_max, y_max), dtype=np.uint8)
    for batch in grouped_batches:
        partial_frames = process_event_batch(batch, min_timestamp, duration, x_max, y_max)
        frames[:partial_frames.shape[0]] += partial_frames

    # Clip values to ensure they don't exceed the max value for uint8
    frames = np.clip(frames, 0, 255)

    return frames

def integrate_events_to_one_frame(df: pd.DataFrame):
    """
    Integrate all DVS camera events into a single frame.

    Parameters:
        df (pd.DataFrame): DataFrame containing event data.
        duration (int): Duration for each frame integration in microseconds.

    Returns:
        np.ndarray: Integrated frame as a numpy array with shape [1, 2, 346, 260].
    """
    # Constants for DAVIS346 sensor resolution
    x_max, y_max = 346, 260

    # Initialize an empty frame with shape [1, 2, x_max, y_max]
    frame = np.zeros((1, 2, x_max, y_max), dtype=np.uint8)

    # Iterate through all events and accumulate them into the single frame
    for _, event in df.iterrows():
        x = event['x']
        y = event['y']
        polarity = event['polarity']
        
        # We accumulate events based on polarity, x, y coordinates
        frame[0, polarity, x, y] += 1

    # Clip values to ensure they don't exceed the max value for uint8
    frame = np.clip(frame, 0, 255)

    return frame

def integrate_events_to_one_frame_1bit(df: pd.DataFrame):
    """
    Integrate all DVS camera events into a single frame.

    Parameters:
        df (pd.DataFrame): DataFrame containing event data.
        duration (int): Duration for each frame integration in microseconds.

    Returns:
        np.ndarray: Integrated frame as a numpy array with shape [1, 2, 346, 260].
    """
    # Constants for DAVIS346 sensor resolution
    x_max, y_max = 346, 260

    # Initialize an empty frame with shape [1, 2, x_max, y_max]
    frame = np.zeros((1, 2, x_max, y_max), dtype=np.bool_)

    # Iterate through all events and accumulate them into the single frame
    for _, event in df.iterrows():
        x = event['x']
        y = event['y']
        polarity = event['polarity']
        
        # We accumulate events based on polarity, x, y coordinates
        frame[0, polarity, x, y] = 1

    return frame

def integrate_events_to_one_frame_1bit_optimized(df: pd.DataFrame):
    """
    Integrate all DVS camera events into a single frame.

    Parameters:
        df (pd.DataFrame): DataFrame containing event data.

    Returns:
        np.ndarray: Integrated frame as a numpy array with shape [1, 2, 346, 260].
    """
    # Constants for DAVIS346 sensor resolution
    x_max, y_max = 346, 260

    # Initialize an empty frame with shape [1, 2, x_max, y_max]
    frame = np.zeros((1, 2, x_max, y_max), dtype=np.bool_)

    # Extract the x, y, and polarity columns as NumPy arrays
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    polarity = df['polarity'].to_numpy()

    # Use NumPy's advanced indexing to set the frame values
    frame[0, polarity, x, y] = 1

    return frame

# Example usage
if __name__ == "__main__":
    csv_file_path = "test\dvSave-2024_09_03_16_05_45_split.csv"  # Replace with the path to your CSV file
    df = pd.read_csv(
        csv_file_path,
        comment='#',  # Skip the header line starting with '#'
    )
    frame_duration = 100  # Example: integrate events every 1000 microseconds
    max_frames = 128  # Example: limit the output to 128 frames

    frames = integrate_events_to_frames(df, frame_duration, max_frames)
    print(f"Integrated frames shape: {frames.shape}")
    np.savez("dvSave-fast3_3000000_1p_" + f"{frame_duration}" + ".npz", frames=frames)  # Save the frames as a .npz file
