import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
from .config import DATA_PROCESSED

MODEL_PATH = os.path.join(DATA_PROCESSED, 'traffic_model.json')

def generate_historical_traffic_data(n_samples=5000):
    """
    Simulates historical trip data.
    Input features: Distance (km), Hour of Day (0-23), Weather Condition (0=Clear, 1=Rain)
    Target: Duration (minutes)
    """
    print("Generating synthetic traffic history...")
    np.random.seed(42)
    
    distances = np.random.uniform(0.5, 25.0, n_samples)
    hours = np.random.randint(6, 22, n_samples) # 6 AM to 10 PM
    weather = np.random.randint(0, 2, n_samples) # 0 or 1
    
    # Logic: Base speed is 30km/h (2 mins per km).
    # Traffic is bad at 9AM and 6PM. Rain adds delay.
    durations = []
    
    for d, h, w in zip(distances, hours, weather):
        base_time = d * 2 # 2 mins per km
        
        # Congestion factor
        congestion = 1.0
        if 8 <= h <= 10 or 17 <= h <= 19: # Peak hours
            congestion = 1.8  # 80% slower
        elif 11 <= h <= 16:
            congestion = 1.2
            
        # Weather factor
        weather_delay = 1.4 if w == 1 else 1.0
        
        # Calculate final time with some random noise
        time = base_time * congestion * weather_delay
        time += np.random.normal(0, 2) # Random noise
        durations.append(max(time, d*1.5)) # Time can't be impossibly fast

    df = pd.DataFrame({
        'distance_km': distances,
        'hour': hours,
        'weather_rain': weather,
        'duration_min': durations
    })
    return df

def train_traffic_model():
    """Trains an XGBoost model to predict travel time."""
    df = generate_historical_traffic_data()
    
    X = df[['distance_km', 'hour', 'weather_rain']]
    y = df['duration_min']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"✅ Traffic Model Trained. MAE: {mae:.2f} minutes")
    
    # Save model
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def predict_time_matrix(distance_matrix, shift_time="09:00", weather_condition=0):
    """
    Converts a Distance Matrix (km) into a Time Matrix (minutes) using the ML model.
    """
    if not os.path.exists(MODEL_PATH):
        train_traffic_model()
        
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    
    # Parse hour from shift_time
    hour = int(shift_time.split(":")[0])
    
    n = distance_matrix.shape[0]
    time_matrix = np.zeros((n, n))
    
    # Flatten matrix for batch prediction (much faster)
    rows, cols = np.indices((n, n))
    flat_distances = distance_matrix.flatten()
    flat_hours = np.full(flat_distances.shape, hour)
    flat_weather = np.full(flat_distances.shape, weather_condition)
    
    input_data = pd.DataFrame({
        'distance_km': flat_distances,
        'hour': flat_hours,
        'weather_rain': flat_weather
    })
    
    # Predict
    flat_times = model.predict(input_data)
    time_matrix = flat_times.reshape((n, n))
    
    # Set diagonal to 0
    np.fill_diagonal(time_matrix, 0)
    
    return time_matrix