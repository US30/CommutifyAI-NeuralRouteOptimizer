from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from .config import VEHICLE_CAPACITY, DEPOT_INDEX

def create_data_model(distance_matrix, demands, num_vehicles):
    """Stores the data for the problem with dynamic vehicle count."""
    data = {}
    data['distance_matrix'] = distance_matrix
    data['demands'] = demands
    data['num_vehicles'] = num_vehicles
    data['vehicle_capacities'] = [VEHICLE_CAPACITY] * num_vehicles
    data['depot'] = DEPOT_INDEX
    return data

def solve_vrp(distance_matrix, df_employees, num_vehicles=4):
    """
    Solves the VRP with a dynamic number of vehicles.
    """
    # 1. Prepare Data
    demands = df_employees['demand'].tolist()
    data = create_data_model(distance_matrix, demands, num_vehicles)

    # 2. Create Routing Index Manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), 
        data['num_vehicles'], 
        data['depot']
    )

    # 3. Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # 4. Create and Register Transit Callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 5. Define Cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 6. Add Capacity Constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],
        True,  # start cumul to zero
        'Capacity'
    )

    # 7. Setting Search Parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # 8. Solve
    solution = routing.SolveWithParameters(search_parameters)

    # 9. Extract Solution
    if solution:
        return extract_solution(data, manager, routing, solution)
    else:
        return None

def extract_solution(data, manager, routing, solution):
    """Extracts the routes from the solution object."""
    all_routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        current_route = []
        route_distance = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            current_route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
        # Add the final return to depot
        node_index = manager.IndexToNode(index)
        # Only add route if it actually moved (has more than just start/end depot)
        if len(current_route) > 1 or route_distance > 0:
            current_route.append(node_index) 
            all_routes.append({
                "vehicle_id": vehicle_id,
                "route": current_route,
                "distance": route_distance
            })
    return all_routes