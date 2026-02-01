import pandas as pd
import numpy as np
from faker import Faker
import os
from .config import DATA_RAW, NUM_EMPLOYEES, CITY_CENTER_LAT, CITY_CENTER_LON, RADIUS_KM

fake = Faker('en_IN')

def generate_synthetic_data():
    print(f"Generating data for {NUM_EMPLOYEES} employees...")
    
    data = []
    for i in range(NUM_EMPLOYEES):
        # Generate random lat/long within a radius approx
        # 1 deg lat ~= 111 km. 0.1 deg ~= 11km.
        lat_offset = np.random.uniform(-0.15, 0.15) 
        lon_offset = np.random.uniform(-0.15, 0.15)
        
        employee = {
            "emp_id": f"EMP_{1000+i}",
            "name": fake.name(),
            "latitude": CITY_CENTER_LAT + lat_offset,
            "longitude": CITY_CENTER_LON + lon_offset,
            "shift_start": np.random.choice(["09:00", "10:00", "14:00"]),
            "demand": 1 # Occupies 1 seat
        }
        data.append(employee)

    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(DATA_RAW, exist_ok=True)
    
    # Add the Office Location as the first row (the Depot)
    office_row = pd.DataFrame([{
        "emp_id": "OFFICE_DEPOT",
        "name": "Corporate HQ",
        "latitude": CITY_CENTER_LAT,
        "longitude": CITY_CENTER_LON,
        "shift_start": "N/A",
        "demand": 0
    }])
    
    df = pd.concat([office_row, df], ignore_index=True)
    
    save_path = os.path.join(DATA_RAW, 'employees.csv')
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    return df