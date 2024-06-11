import math
import pygad
import matplotlib.pyplot as plt


# function to calculate the alloy endurance
def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)


# definition of the gene space
gene_space = [{'low': 0, 'high': 1} for _ in range(6)]


# fitness function
def fitness_func(model, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)


# chromosome and population parameters
num_generations = 100
num_parents_mating = 4
population_size = 50
num_genes = 6
mutation_percent_genes = 15

# running the genetic algorythm
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=population_size,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

# Show the results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution:", solution)
print("Fitness value of the best solution =", solution_fitness)


# Show and save the plot
plt.plot(ga_instance.best_solutions_fitness)
plt.title("Fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig("ex2_plot.jpg")
plt.show()