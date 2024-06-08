import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
import math
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# function to calculate the alloy endurance
def endurance(position):
    x, y, z, u, v, w = position
    return math.exp(-2*(y - math.sin(x))**2) + math.sin(z * u) + math.cos(v * w)

# function for the swarm
def swarm_endurance(swarm):
    return np.array([endurance(p) for p in swarm])

# function for negative alloy endurance
def negative_swarm_endurance(swarm):
    return -swarm_endurance(swarm)

# optimizer parameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

# optimizing the endurance function
best_cost, best_pos = optimizer.optimize(negative_swarm_endurance, iters=1000)

print(f"Best cost: {-best_cost}, Best position: {best_pos}")

# Show the plot
plot_cost_history(optimizer.cost_history)
plt.savefig("ex1_plot.jpg")
plt.show()
