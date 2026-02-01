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
from src.ml_engine import predict_time_matrix
from src.vrp_solver import solve_vrp
from src.config import CITY_CENTER_LAT, CITY_CENTER_LON

# --- HELPER: Time Formatting ---
def format_duration(minutes):
    """Converts minutes to 'X hrs Y mins (Total mins)' format"""
    hours, mins = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours}h {mins}m ({int(minutes)} mins)"
    return f"{mins}m"

# Page Config
st.set_page_config(page_title="Commutify AI", layout="wide")

st.title("🚛 Commutify AI: Intelligent Fleet Routing")
st.markdown("### Machine Learning Powered Route Optimization")

# --- INITIALIZE SESSION STATE ---
if 'data' not in st.session_state:
    try:
        st.session_state['data'] = pd.read_csv("data/raw/employees.csv")
    except:
        st.session_state['data'] = generate_synthetic_data()

if 'routes' not in st.session_state:
    st.session_state['routes'] = None

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Fleet & Traffic Settings")

# 1. Vehicle Count Control
num_vehicles = st.sidebar.slider("🚌 Active Fleet Size", min_value=1, max_value=10, value=4)

# 2. Employee Count
num_employees = st.sidebar.slider("👥 Number of Employees", 10, 100, 30)

# 3. Traffic Condition (Mapped to Time)
traffic_level = st.sidebar.select_slider(
    "🚦 Traffic Condition",
    options=["Low (Midnight)", "Medium (Mid-day)", "High (Rush Hour)"],
    value="High (Rush Hour)"
)

# Map text to hours for the ML model
traffic_map = {
    "Low (Midnight)": "02:00",
    "Medium (Mid-day)": "11:00",
    "High (Rush Hour)": "09:00"
}
shift_time = traffic_map[traffic_level]

# 4. Weather
weather = st.sidebar.radio("Weather", ["Clear", "Rainy"])
weather_val = 1 if weather == "Rainy" else 0

if st.sidebar.button("🔄 Reset / Generate New Data"):
    st.session_state['data'] = generate_synthetic_data()
    st.session_state['routes'] = None
    st.sidebar.success("New Data Generated!")
    st.rerun()

# --- MAIN LOGIC ---

df_display = st.session_state['data'].iloc[:num_employees]

# Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df_display))
col2.metric("Fleet Capacity", f"{num_vehicles} Buses")
col3.metric("Estimated Traffic", traffic_level.split(" (")[0])

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
        
        # Pass num_vehicles dynamically
        routes = solve_vrp(time_matrix, full_df, num_vehicles=num_vehicles)
        
        if routes:
            st.session_state['routes'] = routes
            st.success("Optimization Complete!")
        else:
            st.error("Optimization Failed. Try increasing the Fleet Size.")

# --- VISUALIZATION ---
m = folium.Map(location=[CITY_CENTER_LAT, CITY_CENTER_LON], zoom_start=12)

# Depot Marker
folium.Marker(
    [CITY_CENTER_LAT, CITY_CENTER_LON], 
    tooltip="Office (Depot)",
    icon=folium.Icon(color="black", icon="building", prefix="fa")
).add_to(m)

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

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    for i, vehicle in enumerate(routes):
        route_indices = vehicle['route']
        color = colors[i % len(colors)]
        
        path_coords = []
        for idx in route_indices:
            lat = full_df.iloc[idx]['latitude']
            lon = full_df.iloc[idx]['longitude']
            path_coords.append([lat, lon])
            
            if idx != 0:
                folium.Marker(
                    [lat, lon],
                    tooltip=f"Emp {idx}",
                    icon=folium.Icon(color=color, icon="user", prefix="fa")
                ).add_to(m)

        folium.PolyLine(
            path_coords, color=color, weight=5, opacity=0.8,
            tooltip=f"Vehicle {vehicle['vehicle_id']}"
        ).add_to(m)
        
    st_folium(m, width=1000, height=600)
    
    # --- MANIFEST ---
    st.subheader("📋 Route Manifest")
    
    total_minutes = sum(r['distance'] for r in routes)
    formatted_total = format_duration(total_minutes)
    
    st.info(f"**Total Fleet Time:** {formatted_total}")
    
    for v in routes:
        v_time = format_duration(v['distance'])
        st.text(f"🚌 Vehicle {v['vehicle_id']}: {v_time} | Stops: {len(v['route'])-2}")

else:
    for _, row in df_display.iterrows():
        folium.Marker([row['latitude'], row['longitude']], icon=folium.Icon(color="gray")).add_to(m)
    st_folium(m, width=1000, height=600)