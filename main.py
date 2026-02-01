from src.data_generator import generate_synthetic_data
from src.routing_engine import calculate_distance_matrix

def main():
    # Step 1: Generate Data
    df_employees = generate_synthetic_data()
    
    # Step 2: Create Distance Matrix
    matrix = calculate_distance_matrix(df_employees)
    
    print("\n--- Phase 1 Complete ---")
    print(f"Matrix Shape: {matrix.shape}")
    print(f"Sample Distance (Office to Emp 1): {matrix[0][1]:.2f} km")

if __name__ == "__main__":
    main()