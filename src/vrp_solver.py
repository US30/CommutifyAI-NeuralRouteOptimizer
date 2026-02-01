from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from .config import NUM_VEHICLES, VEHICLE_CAPACITY, DEPOT_INDEX

def create_data_model(distance_matrix, demands):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    data['demands'] = demands
    data['num_vehicles'] = NUM_VEHICLES
    data['vehicle_capacities'] = [VEHICLE_CAPACITY] * NUM_VEHICLES
    data['depot'] = DEPOT_INDEX
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console and returns structured data."""
    print(f"Objective: {solution.ObjectiveValue()} (Total Distance)")
    total_distance = 0
    total_load = 0
    all_routes = []

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for Vehicle {vehicle_id}:\n'
        route_distance = 0
        route_load = 0
        current_route = []
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            
            # Add to route list
            current_route.append(node_index)
            
            plan_output += f' {node_index} Load({route_load}) -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
        node_index = manager.IndexToNode(index)
        plan_output += f' {node_index} Load({route_load})\n'
        plan_output += f'Distance of the route: {route_distance}km\n'
        plan_output += f'Load of the route: {route_load}\n'
        
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
        
        all_routes.append({
            "vehicle_id": vehicle_id,
            "route": current_route,
            "distance": route_distance,
            "load": route_load
        })

    print(f'Total distance of all routes: {total_distance}km')
    print(f'Total load of all routes: {total_load}')
    return all_routes

def solve_vrp(distance_matrix, df_employees):
    print("\n--- Starting VRP Optimization (OR-Tools) ---")
    
    # 1. Prepare Data
    demands = df_employees['demand'].tolist()
    data = create_data_model(distance_matrix, demands)

    # 2. Create Routing Index Manager
    # Arguments: number of locations, number of vehicles, depot node index
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), 
        data['num_vehicles'], 
        data['depot']
    )

    # 3. Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # 4. Create and Register Transit Callback
    # This tells the solver how "expensive" travel is between two nodes
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 5. Define Cost of each arc (Cost = Distance)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 6. Add Capacity Constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

    # 7. Setting Search Parameters
    # We use PATH_CHEAPEST_ARC as a heuristic to find a good first solution quickly
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # 8. Solve
    solution = routing.SolveWithParameters(search_parameters)

    # 9. Output
    if solution:
        return print_solution(data, manager, routing, solution)
    else:
        print("No solution found!")
        return None