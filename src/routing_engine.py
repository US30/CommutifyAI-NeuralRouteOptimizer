import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
from .config import DATA_PROCESSED

def calculate_distance_matrix(df):
    print("Calculating Distance Matrix (Haversine)...")
    coords = df[['latitude', 'longitude']].to_numpy()
    n = len(coords)
    
    # Initialize matrix
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Geodesic calculates distance in Kilometers between (lat, lon) tuples
                dist_matrix[i][j] = geodesic(coords[i], coords[j]).km
            else:
                dist_matrix[i][j] = 0.0
                
    # Save Matrix
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    np.save(os.path.join(DATA_PROCESSED, 'distance_matrix.npy'), dist_matrix)
    print("Distance Matrix saved.")
    return dist_matrix