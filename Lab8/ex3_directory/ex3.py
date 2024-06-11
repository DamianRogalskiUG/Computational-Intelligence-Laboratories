import pygad
import matplotlib.pyplot as plt

labyrinth = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Defining the gene space: 0 - up, 1 - down, 2 - left, 3 - right
gene_space = [0, 1, 2, 3]

# Parameters of the genetic algorithm
population_size = 50
num_generations = 100
num_parents_mating = 10
mutation_percent_genes = 5


# Fitness function
def fitness_func(solution, solution_idx):
    x, y = 1, 1  # Starting position
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

    total_score = 0
    for gene in solution:
        direction = directions[gene]
        new_x, new_y = x + direction[0], y + direction[1]

        # Check if the new position is within the labyrinth boundaries
        if 0 <= new_x < len(labyrinth[0]) and 0 <= new_y < len(labyrinth):
            if labyrinth[new_y][new_x] == 1:  # If it hits the wall
                total_score -= 1  # Penalize for hitting a wall
            else:
                x, y = new_x, new_y
                total_score += 1  # Reward for a valid move
        else:
            total_score -= 1  # Penalize for moving out of boundaries

        # Reward for proximity to the target
        total_score += 1 / (abs(x - 10) + abs(y - 10) + 1) * 10

        if x == 10 and y == 10:  # If reached the target
            total_score += 100  # Maximum reward
            break

    return total_score


# Running the genetic algorithm
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=population_size,
                       num_genes=30,
                       gene_space=gene_space,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=2,
                       stop_criteria=["reach_100"]
                       )

ga_instance.run()

# Show the results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution = {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")

# Plot the fitness over generations
plt.plot(ga_instance.best_solutions_fitness)
plt.title("Fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig("ex3_plot.jpg")
plt.show()
