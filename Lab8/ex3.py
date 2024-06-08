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

# setting the gene space
gene_space = [{'low': 0, 'high': 3} for _ in range(30)]


# setting the parameters of population and parents
population_size = 5
num_generations = 100
num_parents_mating = 2


# set the mutation percent
mutation_percent_genes = 5


# Fitness function
def fitness_func(model, solution, solution_idx):
    x, y = 1, 1  # Starting position
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # available moves

    total_score = 0
    for gene in solution:
        direction = directions[int(gene)]
        new_x, new_y = x + direction[0], y + direction[1]
        if labyrinth[new_y][new_x] == 1:  # If met the wall
            total_score -= 1  # Penalize for hitting a wall
        else:
            x, y = new_x, new_y
            total_score += 1  # Reward for valid move

        # Reward for proximity to the target
        total_score += 1 / (abs(x - 10) + abs(y - 10) + 1) * 10

        if x == 10 and y == 10:  # If finished the labyrinth
            total_score += 100  # Maximum reward
            break

    return total_score


# running the genetic algorythm
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

# show the results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution = {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")


# Show and save the plot
plt.plot(ga_instance.best_solutions_fitness)
plt.title("Fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig("ex3_plot.jpg")
# plt.show()