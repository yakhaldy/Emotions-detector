import numpy as np
import pandas as pd

def load_data(file_path):
    try:
        print(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    


def preprocess_pixels(pixels):
    try:
       return np.array([
                np.fromstring(p, sep=" ", dtype="float32").reshape(48, 48, 1) / 255.0
                for p in pixels
            ])
    except Exception as e:
        print(f"Error preprocessing pixels: {e}")
        return None
