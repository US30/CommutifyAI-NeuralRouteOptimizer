import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, '../data/raw')
DATA_PROCESSED = os.path.join(BASE_DIR, '../data/processed')

# Simulation Settings
NUM_EMPLOYEES = 100
CITY_CENTER_LAT = 12.9716  # Bangalore
CITY_CENTER_LON = 77.5946
RADIUS_KM = 15  # Employees live within 15km of office

# Fleet Settings (Phase 2)
NUM_VEHICLES = 4
VEHICLE_CAPACITY = 15  # Each bus can take 15 people
DEPOT_INDEX = 0        # The office is at index 0 (row 0)