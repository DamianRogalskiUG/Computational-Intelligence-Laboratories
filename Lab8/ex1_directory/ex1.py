import numpy as np
import pygad
import time

from matplotlib import pyplot as plt

# Data for the problem
items_dict = [
    {"name": "zegar", "value": 100, "weight": 7},
    {"name": "obraz-pejzaż", "value": 300, "weight": 7},
    {"name": "obraz-portret", "value": 200, "weight": 6},
    {"name": "radio", "value": 40, "weight": 2},
    {"name": "laptop", "value": 500, "weight": 5},
    {"name": "lampka nocna", "value": 70, "weight": 6},
    {"name": "srebrne sztućce", "value": 100, "weight": 1},
    {"name": "porcelana", "value": 250, "weight": 3},
    {"name": "figura z brązu", "value": 300, "weight": 10},
    {"name": "skórzana torebka", "value": 280, "weight": 3},
    {"name": "odkurzacz", "value": 300, "weight": 15},
]
max_capacity = 25

# Fitness function
def fitness_func(model, solution, solution_idx):
    total_weight = 0
    total_value = 0

    for idx, item in enumerate(items_dict):
        if solution[idx] == 1:
            total_value += item["value"]
            total_weight += item["weight"]

    # Give penalty for exceeding the capacity limit
    if total_weight > max_capacity:
        return 0
    return total_value

# Genetic Algorithm parameters
sol_per_pop = 100
num_genes = len(items_dict)
num_parents_mating = 5
num_generations = 100  # Number of generations
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10  # Lower mutation rate to prevent excessive randomness

# List to store the value results
best_solutions_values = []

# Counter for successful attempts
successful_attempts = 0
total_time = 0
plot_counter = 0

# Run the GA 10 times to measure success rate and average time
for i in range(10):
    start = time.time()  # Start time measurement

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_space=[0, 1])

    # Run the GA
    ga_instance.run()

    # End time measurement
    end = time.time()
    elapsed_time = end - start
    total_time += elapsed_time

    # Summary
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solutions_values.append((solution, solution_fitness))
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


    # Check if the solution is the optimal one
    if solution_fitness == 1630 or solution_fitness >= 1000:
        successful_attempts += 1

    # Show and save the plot
    plt.plot(ga_instance.best_solutions_fitness)
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(f"plots/ex1_plot{plot_counter}.jpg")
    plt.show()

    # Increment plot counter for plot images
    plot_counter += 1

# e) Success rate of the algorithm
success_rate = successful_attempts / 10 * 100

# f) Average runtime of the algorithm
average_time = total_time / 10

print(f"\nSuccess rate of the algorithm: {success_rate}%")
print(f"Average runtime of the algorithm: {average_time:.6f} seconds")
