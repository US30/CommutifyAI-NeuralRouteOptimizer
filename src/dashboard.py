import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import sys
import os

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import generate_synthetic_data
from src.routing_engine import calculate_distance_matrix
from src.ml_engine import predict_time_matrix, train_traffic_model
from src.vrp_solver import solve_vrp
from src.config import CITY_CENTER_LAT, CITY_CENTER_LON

# Page Config
st.set_page_config(page_title="Commutify AI", layout="wide")

st.title("🚛 Commutify AI: Intelligent Fleet Routing")
st.markdown("### Machine Learning Powered Route Optimization")

# --- INITIALIZE SESSION STATE ---
# This acts as the "Memory" for the app
if 'data' not in st.session_state:
    try:
        # Load existing data or generate new
        st.session_state['data'] = pd.read_csv("data/raw/employees.csv")
    except:
        st.session_state['data'] = generate_synthetic_data()

if 'routes' not in st.session_state:
    st.session_state['routes'] = None  # No routes calculated yet

# --- SIDEBAR ---
st.sidebar.header("⚙️ Simulation Parameters")
# We use keys so the slider updates the state directly
num_employees = st.sidebar.slider("Number of Employees", 10, 100, 30)
shift_time = st.sidebar.selectbox("Shift Start Time", ["09:00", "14:00", "22:00", "06:00"])
weather = st.sidebar.radio("Weather Condition", ["Clear", "Rainy"])
weather_val = 1 if weather == "Rainy" else 0

if st.sidebar.button("🔄 Generate New Data"):
    st.session_state['data'] = generate_synthetic_data()
    st.session_state['routes'] = None # Reset routes since data changed
    st.sidebar.success("New Data Generated! Click Optimize to route.")
    st.rerun() # Force a refresh

# --- MAIN LOGIC ---

# Filter data based on slider
df_display = st.session_state['data'].iloc[:num_employees]

# Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df_display))
col2.metric("Active Vehicles", "4")
col3.metric("Traffic Condition", "Heavy" if shift_time in ["09:00", "18:00"] else "Moderate")

# --- BUTTON LOGIC ---
if st.button("🚀 Optimize Routes"):
    with st.spinner('Calculating Matrices & Optimizing...'):
        # Prepare Data
        office_row = pd.DataFrame([{
            "emp_id": "OFFICE_DEPOT",
            "latitude": CITY_CENTER_LAT,
            "longitude": CITY_CENTER_LON,
            "demand": 0
        }])
        full_df = pd.concat([office_row, df_display], ignore_index=True)

        # Execute Engine
        dist_matrix = calculate_distance_matrix(full_df)
        time_matrix = predict_time_matrix(dist_matrix, shift_time, weather_val)
        
        # Save result to Session State
        routes = solve_vrp(time_matrix, full_df)
        if routes:
            st.session_state['routes'] = routes
            st.success("Optimization Complete!")
        else:
            st.error("Optimization Failed.")

# --- VISUALIZATION LOGIC ---
# This runs on every refresh, checking if 'routes' exists in memory

m = folium.Map(location=[CITY_CENTER_LAT, CITY_CENTER_LON], zoom_start=12)

# Always plot Depot
folium.Marker(
    [CITY_CENTER_LAT, CITY_CENTER_LON], 
    tooltip="Office (Depot)",
    icon=folium.Icon(color="black", icon="building", prefix="fa")
).add_to(m)

# CHECK: Do we have optimized routes in memory?
if st.session_state['routes']:
    routes = st.session_state['routes']
    
    # Re-prepare full DF for plotting
    office_row = pd.DataFrame([{
        "emp_id": "OFFICE_DEPOT",
        "latitude": CITY_CENTER_LAT,
        "longitude": CITY_CENTER_LON,
        "demand": 0
    }])
    full_df = pd.concat([office_row, df_display], ignore_index=True)

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for vehicle in routes:
        route_indices = vehicle['route']
        vehicle_id = vehicle['vehicle_id']
        color = colors[vehicle_id % len(colors)]
        
        path_coords = []
        for idx in route_indices:
            lat = full_df.iloc[idx]['latitude']
            lon = full_df.iloc[idx]['longitude']
            path_coords.append([lat, lon])
            
            # Plot Employee Marker (Skip Depot 0)
            if idx != 0:
                folium.Marker(
                    [lat, lon],
                    tooltip=f"Emp {idx}",
                    icon=folium.Icon(color=color, icon="user", prefix="fa")
                ).add_to(m)

        # Draw Line
        folium.PolyLine(
            path_coords, 
            color=color, 
            weight=5, 
            opacity=0.8,
            tooltip=f"Vehicle {vehicle_id}"
        ).add_to(m)
        
    st_folium(m, width=1000, height=600)
    
    # Show Manifest
    st.subheader("📋 Route Manifest")
    total_time = sum(r['distance'] for r in routes)
    st.info(f"Total Fleet Time: {total_time:.0f} minutes")
    
    for v in routes:
        st.text(f"🚌 Vehicle {v['vehicle_id']}: {v['distance']} mins | Stops: {len(v['route'])-2}")

else:
    # No routes yet? Just show grey markers
    for _, row in df_display.iterrows():
        folium.Marker([row['latitude'], row['longitude']], icon=folium.Icon(color="gray")).add_to(m)
    st_folium(m, width=1000, height=600)