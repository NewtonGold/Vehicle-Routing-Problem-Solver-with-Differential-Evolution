import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import sys
import pickle

np.set_printoptions(threshold=sys.maxsize)

# Define problem parameters
# instance_name = "P-n16-k8"
instance_name = "A-n32-k5"
with open("./cvrp_instances/" + instance_name + ".pkl", 'rb') as file:
    instance = pickle.load(file)
with open("./cvrp_solutions/" + instance_name + ".pkl", 'rb') as file:
    global_solution = pickle.load(file)
print(instance)
print(global_solution)

NUM_CUSTOMERS = instance['dimension'] - 1
NUM_VEHICLES = int(instance_name.split('k')[1])
CAPACITY = instance['capacity']
DE_POPULATION_SIZE = 10
DE_MAX_GENERATIONS = 2500
DE_CR = 0.1
DE_F = 0.7

def main():
    # Generate random customer demands
    np.random.seed(42)
    # customer_demands = np.random.randint(1, 3, size=NUM_CUSTOMERS)
    # customer_demands = np.concatenate(([0], customer_demands))
    # print('Customer Demands: ', customer_demands)

    # # Define distance matrix
    # locations = np.random.rand(NUM_CUSTOMERS + 1, 2)
    # distances = np.zeros((NUM_CUSTOMERS + 1, NUM_CUSTOMERS + 1))
    # for i in range(NUM_CUSTOMERS + 1):
    #     for j in range(i + 1, NUM_CUSTOMERS + 1):
    #         distances[i, j] = np.linalg.norm(locations[i] - locations[j])
    #         distances[j, i] = distances[i, j]
    if 'node_coord' in instance:
        locations = instance['node_coord']
    else:
        locations = np.random.rand(NUM_CUSTOMERS + 1, 2)
    # locations = instance['node_coord']
    print('Locations: ', locations)
    customer_demands = instance['demand']
    distances = np.rint(instance['edge_weight'])

    # print('instance: ', instance)

    # print(solution)

    # print('Distances: ', distances)
    # print('Customer Demands: ', customer_demands)

    
    # trial_solution = np.array([0, 21, 31, 19, 17, 13, 7, 26, 0, 12, 1, 16, 30, 0, 27, 24, 0, 29, 18, 8, 9, 22, 15, 10, 25, 5, 20, 0, 14, 28, 11, 4, 23, 3, 2, 6, 0])

    # # Plot graph
    # plot_graph(customer_demands, locations, distances)

    solution_array = [0]
    for route in global_solution['routes']:
        route = np.concatenate((route, [0]))
        solution_array += route.tolist()
    solution_array = np.array(solution_array)
    print('Solution Array: ', solution_array)
    total_distance = fitness(solution_array, distances, customer_demands)
    print('The total distance is: ', total_distance)

    # # Check Validity Function
    # trial_solution = np.array([0, 5, 1, 0, 2, 0, 4, 3, 0])
    # valid_solution = check_validity(trial_solution, customer_demands)
    # print('Valid Solution: ', valid_solution)

    # Run Differential Evolution algorithm
    best_solution, best_fitness, fitness_history = differential_evolution(
        fitness_fn=fitness,
        population_size=DE_POPULATION_SIZE,
        max_generations=DE_MAX_GENERATIONS,
        crossover_rate=DE_CR,
        f=DE_F,
        customer_demands=customer_demands,
        distances=distances
    )
        
    # # Print best solution and fitness
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    validity = check_validity(best_solution, customer_demands)
    print('Valid Solution: ', validity)

    # # # Plot graph with solution
    # graph_solution(customer_demands, locations, distances, best_solution)
    plot_cvrp_solution(locations, best_solution)

    # Visualize fitness history
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness History")
    plt.show()

    # # Visualize solution
    # routes = np.split(best_solution, np.where(best_solution == 0)[0])
    # routes = [r for r in routes if len(r) > 0]
    # colors = ["r", "g", "b", "c", "m", "y", "k"]
    # fig, ax = plt.subplots()
    # for i, r in enumerate(routes):
    #     x = locations[r, 0]
    #     y = locations[r, 1]
    #     ax.scatter(x[1:-1], y[1:-1], color=colors[i % len(colors)])
    #     ax.plot(x, y, color=colors[i % len(colors)])
    # ax.scatter(locations[0, 0], locations[0, 1], marker="s", color="k")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_title("Best Solution")
    # plt.show()

# Define fitness function
def fitness(solution, distances, customer_demands):
    """
    Computes the total distance traveled by all vehicles in the solution, given a list of routes.
    """
    total_distance = 0
    routes = np.split(solution, np.where(solution == 0)[0])
    routes = [r for r in routes if len(r) > 0]
    # total_distance = 0
    for r in routes:
        r = np.concatenate((r, [0]))
        route_distance = 0
        # print(r)
        for i in range(len(r) - 1):
            if r[i] < len(distances) and r[i] >= 0 and r[i+1] < len(distances) and r[i+1] >= 0:
                route_distance += distances[r[i]][r[i + 1]]
        total_distance += route_distance

    # Check validity of solution
    valid_solution = check_validity(solution, customer_demands)
    total_distance += ((np.sum(distances) * len(distances)) / np.count_nonzero(distances)) * valid_solution

    return total_distance

# Define Differential Evolution algorithm
def differential_evolution1(fitness_fn, population_size, max_generations, crossover_rate, f, customer_demands):
    """
    Implements the Differential Evolution algorithm for optimizing a given fitness function.
    """
    population = np.random.randint(NUM_CUSTOMERS + 1, size=(population_size, NUM_CUSTOMERS + NUM_VEHICLES))
    best_fitness = 0
    print('Best Fitness: ', best_fitness)
    for solution in population:
        fitness = fitness_fn(solution, customer_demands)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = solution
    # best_fitness = np.inf
    # best_solution = None
    fitness_history = []
    for i in range(max_generations):
        for j in range(population_size):
            # Select three random individuals from the population
            a, b, c = population[np.random.choice(population_size, size=3, replace=False)]
            # Perform crossover operation
            mask = np.random.rand(NUM_CUSTOMERS + NUM_VEHICLES) < crossover_rate
            print('Population_j : ', population[j])
            print('Mask: ', mask)
            trial_solution = np.where(mask, a + f * (b - c), population[j])
            print('Trial Solution: ', trial_solution)
            
            # Ensure that the trial solution is valid
            print(np.clip(trial_solution, 0, NUM_CUSTOMERS))
            trial_solution = np.unique(np.clip(trial_solution, 0, NUM_CUSTOMERS)).astype(int)
            print('Trial Solution: ', trial_solution)
            trial_solution = trial_solution[trial_solution < NUM_CUSTOMERS]
            print(np.sum(customer_demands[trial_solution]))
            if np.sum(customer_demands[trial_solution]) > CAPACITY:
                trial_solution = population[j]
                
            # Evaluate fitness of trial solution
            trial_fitness = fitness_fn(trial_solution)
            # Update population with trial solution if it is better
            #print('Trial Fitness: ' + str(trial_fitness) + ', Population Fitness: '  + str(fitness_fn(population[j])))
            if trial_fitness < fitness_fn(population[j]):
                #print('Population before:', population[j])
                #print('Trial solution: ', trial_solution)
                population[j] = trial_solution
                #print('Population after: ', population[j])
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_solution
        fitness_history.append(best_fitness)
        # print(f"Generation {i + 1}/{max_generations}: Best fitness = {best_fitness}")
    return best_solution, best_fitness, fitness_history

# Define Differential Evolution algorithm
def differential_evolution(fitness_fn, population_size, max_generations, crossover_rate, f, customer_demands, distances):
    # Initialize population
    population = generate_population(population_size)
    # solution_array = [0]
    # for route in global_solution['routes']:
    #     route = np.concatenate((route, [0]))
    #     solution_array += route.tolist()
    # solution_array = np.array(solution_array)
    # print('Solution Array: ', solution_array)
    # population = np.append(population, [solution_array], axis=0)
    # print('Population: ', population)
    population_fitness = np.array([fitness_fn(solution, distances, customer_demands) for solution in population])
    # print('Population Fitness: ', population_fitness)

    best_fitness = np.inf
    for solution in population:
        # if np.array_equal(solution, solution_array):
        #     pass
        fitness = fitness_fn(solution, distances, customer_demands)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution
            
    fitness_history = []
    for i in range(max_generations):
        for j in range(population_size):
            # Perform mutation and crossover
            trial_solution = perform_mutation(population.copy(), population_size, j, f, crossover_rate)
            # print('Trial Solution: ', trial_solution)
            # Check validity of trial solution
            # valid_solution = check_validity(trial_solution, customer_demands)
            # # print('Valid Solution: ', valid_solution)
            # if valid_solution:
            # Evaluate fitness of trial solution
            trial_fitness = fitness_fn(trial_solution, distances, customer_demands)
                
            # Update population with trial solution if it is better
            population, best_fitness, best_solution = update_population(population, j, trial_solution, trial_fitness, best_fitness, best_solution, fitness_fn, distances, customer_demands)
        
        fitness_history.append(best_fitness)
        if i % 100 == 0:
            print(f"Generation {i + 1}/{max_generations}: Best fitness = {best_fitness}")
        # print('Population: ', population)
        # population_fitness = np.array([fitness_fn(solution, distances, customer_demands) for solution in population])
        # print('Population Fitness: ', population_fitness)
    return best_solution, best_fitness, fitness_history

def generate_population(population_size):
    # Generate random population, where each individual is a list of customer IDs
    # This initial population does not include the depot (customer 0)
    # population = np.random.randint(1, NUM_CUSTOMERS, size=(population_size, NUM_CUSTOMERS))
    population = np.zeros((population_size, NUM_CUSTOMERS), dtype=int)
    for i in range(population_size):
        population[i] = np.arange(1, NUM_CUSTOMERS + 1)

    # Add the number of returns to the depot so the specified number of vehicles is used
    vehicles = np.zeros((population_size, NUM_VEHICLES - 1), dtype=int)
    population = np.concatenate((population, vehicles), axis=1)

    # Shuffle the location where each vehicle returns to the depot
    for solution in population:
        solution = np.random.shuffle(solution)
    zeros = np.zeros((population_size, 1), dtype=int)

    # Add the depot to the beginning and end of each solution
    population = np.concatenate((population, zeros), axis=1)
    population = np.concatenate((zeros, population), axis=1)

    return population

def perform_mutation(population, population_size, current_idx, f, crossover_rate):
    # Select three random individuals from the population
    a_idx, b_idx, c_idx = np.random.choice(population_size, size=3, replace=False)
    a, b, c = population[a_idx], population[b_idx], population[c_idx]

    # Select current population member
    current_pop = population[current_idx].copy()
    current_pop = np.delete(current_pop, [0])
    current_pop = np.delete(current_pop, [-1])

    # Set up trial solution
    trial_solution = np.zeros(len(current_pop), dtype=int)

    # Perform crossover operation
    chosen = np.random.randint(NUM_CUSTOMERS + NUM_VEHICLES - 2)
    values = np.arange(NUM_CUSTOMERS + NUM_VEHICLES - 2)
    count = 0
    swapped = []
    for k in range(len(current_pop)):
        if k not in swapped:
            if np.random.rand() <= crossover_rate or k == chosen:
                if len(values) != 0:
                    value_index = int((a[k] + f * (b[k] - c[k])) % (len(values)))
                else:
                    break
                # print('Value Index: ', value_index)
                swap_index = values[value_index]
                trial_solution[k] = current_pop[swap_index]
                trial_solution[swap_index] = current_pop[k]
                swapped.append(swap_index)
                swapped.append(k)
                values = np.delete(values, value_index)
                values = np.delete(values, np.where(values == k))
            else:
                trial_solution[k] = current_pop[k]

    trial_solution = np.concatenate(([0], trial_solution))
    trial_solution = np.concatenate((trial_solution, [0]))
    return trial_solution

def check_validity(trial_solution, customer_demands):
    valid_solution = 0
    start_position = 0
    temp = trial_solution[trial_solution != 0].copy()
    unique = np.unique(temp)
    if len(unique) != len(temp):
        valid_solution = False
    elif trial_solution.max() > NUM_CUSTOMERS:
        valid_solution = False
    elif trial_solution.min() < 0:
        valid_solution = False
    else: 
        for _ in range(NUM_VEHICLES):
            route_start = None
            route_end = None
            count = 0
            for l in range(start_position, NUM_CUSTOMERS + NUM_VEHICLES + 1):
                if trial_solution[l] == 0 and route_start is None:
                    route_start = l
                elif trial_solution[l] == 0 and route_start is not None:
                    route_end = l
                    break
                else:
                    count += 1
            if route_start is None or route_end is None or count == 0:
                valid_solution += 1
                # break
            route = trial_solution[route_start:route_end+1]
            if np.sum(np.fromiter([customer_demands[i] for i in route[route != 0]], float)) > CAPACITY:
                valid_solution += 1
                # break
            start_position = route_end
        if start_position != len(trial_solution) - 1:
            valid_solution += 1
    return valid_solution

def update_population(population, current_idx, trial_solution, trial_fitness, best_fitness, best_solution, fitness_fn, distances, customer_demands):
    # Update population with trial solution if it is better
    # print('Trial Fitness: ', trial_fitness)
    # print('Current Solution Fitness: ', fitness_fn(population[current_idx], distances, customer_demands))
    # print(trial_solution)
    if trial_fitness < fitness_fn(population[current_idx], distances, customer_demands):
        # duplicate = False
        # for i in range(len(population)):
        #     if np.array_equal(population[i], trial_solution):
        #         duplicate = True
        # if not duplicate:
        population[current_idx] = trial_solution
        if trial_fitness < best_fitness:
            best_fitness = trial_fitness
            best_solution = trial_solution
    return population, best_fitness, best_solution

# def plot_original_graph()
#     import networkx as nx
#     import matplotlib.pyplot as plt

#     # Define plot size
#     width = 20
#     height = 12

#     # Create graph with locations as nodes
#     G = nx.Graph()
#     for i in range(NUM_CUSTOMERS + 1):
#         demand = customer_demands[i-1] if i > 0 else 0
#         G.add_node(i, pos=locations[i], demand=demand)

#     # Add edges between nodes with non-zero distances
#     for i in range(NUM_CUSTOMERS + 1):
#         for j in range(i + 1, NUM_CUSTOMERS + 1):
#             if distances[i, j] > 0:
#                 G.add_edge(i, j, weight=round(distances[i, j], 2))

#     # Draw graph with node positions and edge weights
#     #pos = nx.get_node_attributes(G, 'pos')
#     #edge_labels = nx.get_edge_attributes(G, 'weight')
#     #plt.figure(figsize=(width, height))
#     #nx.draw(G, pos, with_labels=True)
#     #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     # Draw graph with node positions, demand, and edge weights
#     pos = nx.get_node_attributes(G, 'pos')
#     demands = nx.get_node_attributes(G, 'demand')
#     labels = {i: f'{i} ({demands[i]})' for i in G.nodes}
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     plt.figure(figsize=(width, height))
#     nx.draw(G, pos, with_labels=False, node_size=800)
#     nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_family='sans-serif')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#     # Add legend with node IDs and coordinates
#     legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', markersize=8, markerfacecolor='black', label=f'Node {i} ({x}, {y})') for i, (x, y) in pos.items()]
#     plt.legend(handles=legend_handles, loc='lower left', title='Node Locations')
#     plt.show()

def graph_solution(customer_demands, locations, distances, solution):   
    # Define plot size
    # width = 20
    # height = 12

    # Create graph with locations as nodes
    G = nx.Graph()
    for i in range(NUM_CUSTOMERS + 1):
        demand = customer_demands[i-1] if i > 0 else 0
        G.add_node(i, pos=locations[i], demand=demand)

    # Add edges between nodes with non-zero distances
    # for i in range(NUM_CUSTOMERS + 1):
    #     for j in range(i + 1, NUM_CUSTOMERS + 1):
    #         if distances[i, j] > 0:
    #             G.add_edge(i, j, weight=round(distances[i, j], 2))

    pos = nx.get_node_attributes(G, 'pos')
    demands = nx.get_node_attributes(G, 'demand')
    labels = {i: f'{i} ({demands[i]})' for i in G.nodes}
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # plt.figure(figsize=(width, height))
    nx.draw(G, pos, with_labels=False, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Plot Solution Vehicle Routes
    routes = np.split(solution, np.where(solution == 0)[0])
    routes.pop(0)
    routes.pop(-1)
    
    # Define a color map to use for the routes
    cmap = mpl.colormaps['hsv']

    # Create a dictionary to store the colors for each route
    color_dict = {}

    for i in range(len(routes)):
        route = routes[i]
        route = np.concatenate((route, [0]))
        edges = [(route[j], route[j+1]) for j in range(len(route)-1)]
        edge_labels = {(route[j], route[j+1]): distances[route[j], route[j+1]] for j in range(len(route)-1)}
        print(edges)
        for edge in edges.copy():
            edges.append((edge[1], edge[0]))
            edge_labels[(edge[1], edge[0])] = edge_labels[(edge[0], edge[1])]
        color = cmap(i / len(routes))
        nx.draw_networkx_edges(G, pos, edgelist=edges, label=edge_labels, width=3.0, edge_color=color)
        # Add the color for this route to the color dictionary
        color_dict[f"Route {i}"] = color

    # Add legend with node IDs and coordinates
    legend_handles = [Patch(facecolor=color_dict[label], label=label) for label in color_dict.keys()]
    plt.legend(handles=legend_handles, loc='lower left', title='Routes')
    plt.show()

def plot_cvrp_solution(locations, solution):
    
    # # Define a color map to use for the routes
    cmap = mpl.colormaps['hsv']

    # # Create a dictionary to store the colors for each route
    color_dict = {}
    
    # # Create a plot and set the plot size
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # # Plot the customer locations
    ax.scatter([loc[0] for loc in locations], [loc[1] for loc in locations], s=100, color='black')

    # # Get the solution vehicle routes
    routes = np.split(solution, np.where(solution == 0)[0])
    routes.pop(0)
    routes.pop(-1)
    
    # # Plot the solution routes
    for i in range(len(routes)):
        route = routes[i]
        route = np.concatenate((route, [0]))

        color = cmap(i / len(routes))
        
        # # Create a line plot for the route
        ax.plot([locations[x][0] for x in route], [locations[x][1] for x in route], color=color, linewidth=3, label=f'Vehicle {i}')

        color_dict[f"Route {i}"] = color
        
    
    # # Set the axis limits and labels
    ax.set_xlim([0, max([loc[0] for loc in locations]) * 1.1])
    ax.set_ylim([0, max([loc[1] for loc in locations]) * 1.1])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # # Set the title
    ax.set_title(f'CVRP Solution ({NUM_VEHICLES} Vehicles, Capacity {CAPACITY})')

    # # Create a legend for the solution routes
    legend_handles = [Patch(facecolor=color_dict[label], label=label) for label in color_dict.keys()]

    # # Define the coordinates for the legend box
    legend_x = 1
    legend_y = 0.5
    
    # # Place the legend box outside of the graph area
    plt.legend(handles=legend_handles, bbox_to_anchor=(legend_x, legend_y), loc='center left', title='Routes')
    
    # # Show the plot
    plt.show()

if __name__ == '__main__':
    main()