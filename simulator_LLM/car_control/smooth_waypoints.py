import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# --- Configuration ---
INPUT_DIR = 'car_control/CSVs'
OUTPUT_DIR = '.' # Current directory
# Files to process (center, inner, outer lanes)
TARGET_FILES = ['Centerline_points.csv', 'inner_bound_points.csv', 'outer_bound_points.csv']

# Savitzky-Golay filter parameters
# Note: window_length must be an odd integer and greater than polyorder.
# A larger window_length produces a smoother curve.
WINDOW_LENGTH = 11
POLY_ORDER = 3

# --- Main Script ---
def smooth_csv_files():
    """
    Reads target CSV files, applies a Savitzky-Golay filter to smooth the
    x, y coordinates, and saves them to new files.
    """
    print(f"Looking for CSV files in: {os.path.abspath(INPUT_DIR)}")

    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
        return

    for filename in TARGET_FILES:
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = f"smoothed_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if not os.path.exists(input_path):
            print(f"Warning: File not found, skipping: {input_path}")
            continue

        try:
            print(f"Processing: {input_path}")
            # Read CSV without headers, assuming two columns [x, y]
            df = pd.read_csv(input_path, header=None, names=['x', 'y'])

            if df.shape[1] < 2:
                print(f"Warning: Skipping {filename}, expected at least 2 columns.")
                continue

            # Ensure we have enough data points for the filter
            if len(df) < WINDOW_LENGTH:
                print(f"Warning: Not enough data points in {filename} to apply smoothing (need at least {WINDOW_LENGTH}). Skipping.")
                continue

            # Apply Savitzky-Golay filter to x and y columns
            # mode='wrap' is suitable for closed-loop tracks to handle the start/end points smoothly.
            df['x_smooth'] = savgol_filter(df['x'], window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, mode='wrap')
            df['y_smooth'] = savgol_filter(df['y'], window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, mode='wrap')

            # Create a new dataframe with only the smoothed data
            smoothed_df = df[['x_smooth', 'y_smooth']]

            # Save the smoothed data to a new CSV file, without header and index
            smoothed_df.to_csv(output_path, header=False, index=False)
            print(f"Successfully created smoothed file: {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

if __name__ == '__main__':
    # Note: This script requires the SciPy library.
    # You can install it with: pip install scipy
    smooth_csv_files()
