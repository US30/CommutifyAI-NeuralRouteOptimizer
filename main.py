from src.data_generator import generate_synthetic_data
from src.routing_engine import calculate_distance_matrix
from src.vrp_solver import solve_vrp
from src.ml_engine import predict_time_matrix, train_traffic_model

def main():
    print("--- 🚀 Commutify AI Initialization ---")
    
    # Step 1: Generate Data
    df_employees = generate_synthetic_data()
    
    # Step 2: Distance Matrix
    dist_matrix = calculate_distance_matrix(df_employees)
    
    # Step 3: Train/Load ML Model
    print("\n--- 🧠 Phase 3: Traffic AI Engine ---")
    train_traffic_model()
    
    # Step 4: Predict Time Matrix (e.g., for a 9 AM Shift with Rain)
    print("\nPredicting traffic for 9:00 AM Rush Hour...")
    time_matrix = predict_time_matrix(dist_matrix, shift_time="09:00", weather_condition=0)
    
    # Step 5: Solve VRP (Optimize for TIME, not Distance)
    print("\n--- 🔄 Optimizing Routes for TIME (Minimize Duration) ---")
    routes = solve_vrp(time_matrix, df_employees)

    if routes:
        print("\n✅ AI Optimization Complete.")
    else:
        print("\n❌ Optimization Failed")

if __name__ == "__main__":
    main()