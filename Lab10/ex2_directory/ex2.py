import numpy as np
import pygad
import gym
from matplotlib import pyplot as plt

# Env settings
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="ansi")
env.reset()

# GA parameters
num_generations = 20
num_parents_mating = 2
sol_per_pop = 20
num_genes = 30
gene_space = [0, 1, 2, 3]  # 0: left, 1: down, 2: right, 3: up

# Fitness function
def fitness_func(model, solution, solution_idx):
    env.reset()
    initial_state = env.s
    total_reward = 0
    for action in solution:
        observation, reward, terminated, truncated, info = env.step(int(action))
        if observation != initial_state:
            total_reward += 1 # reward for getting closer to the target
        total_reward += reward
        if terminated:
            total_reward -= reward # penalty for getting of the target
            break
    return total_reward

# GA instance
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       mutation_percent_genes=10,
                       mutation_type="random",
                       parent_selection_type="tournament",
                       crossover_type="two_points")

# Running the algorythm
ga_instance.run()

# Show the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


env.reset()
for action in solution:
    env.render()
    observation, reward, terminated, truncated, info = env.step(int(action))
    if terminated:
        break
env.close()

# Show and save the plot
plt.plot(ga_instance.best_solutions_fitness)
plt.title("Fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig("ex2_plot.jpg")
plt.show()