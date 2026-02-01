from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import uvicorn

# Import your modules
from .routing_engine import calculate_distance_matrix
from .ml_engine import predict_time_matrix, train_traffic_model
from .vrp_solver import solve_vrp
from .config import CITY_CENTER_LAT, CITY_CENTER_LON

app = FastAPI(title="Commutify AI API", version="1.0")

# --- Pydantic Models (Data Validation) ---
class EmployeeRequest(BaseModel):
    emp_id: str
    latitude: float
    longitude: float
    demand: int = 1

class RouteRequest(BaseModel):
    employees: List[EmployeeRequest]
    shift_time: str = "09:00"
    weather_rain: int = 0  # 0 or 1

# --- API Endpoints ---

@app.get("/")
def health_check():
    return {"status": "active", "system": "Neural Route Optimizer"}

@app.post("/optimize")
def optimize_routes(payload: RouteRequest):
    """
    Accepts a list of employee locations and returns optimized routes
    considering traffic predictions.
    """
    try:
        # 1. Convert Input JSON to DataFrame
        data = [e.dict() for e in payload.employees]
        df = pd.DataFrame(data)
        
        # Add Depot (Office) manually as index 0
        office_row = pd.DataFrame([{
            "emp_id": "OFFICE_DEPOT",
            "latitude": CITY_CENTER_LAT,
            "longitude": CITY_CENTER_LON,
            "demand": 0
        }])
        df = pd.concat([office_row, df], ignore_index=True)

        # 2. Pipeline Execution
        print("⚡ Request Received. Calculating Matrix...")
        dist_matrix = calculate_distance_matrix(df)
        
        print("🔮 Predicting Traffic...")
        time_matrix = predict_time_matrix(
            dist_matrix, 
            shift_time=payload.shift_time, 
            weather_condition=payload.weather_rain
        )
        
        print("🧩 Solving VRP...")
        routes = solve_vrp(time_matrix, df)
        
        if not routes:
            raise HTTPException(status_code=400, detail="Optimization failed to find a solution.")
            
        return {
            "status": "success", 
            "total_time_min": sum(r['distance'] for r in routes),
            "routes": routes
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure traffic model exists before starting
    train_traffic_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)