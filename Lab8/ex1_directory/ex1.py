import numpy as np
import pygad
import time

# Dane problemu
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


# Funkcja fitness
def fitness_func(model, solution, solution_idx):
    total_weight = 0
    total_value = 0

    for solution_idx, item in enumerate(items_dict):
        if solution[solution_idx] == 1:
            total_value += item['value']
            total_weight += item['weight']

    return total_value  # Zwracamy wartość jako fitness

# Parametry algorytmu genetycznego
sol_per_pop = 100
num_genes = len(items_dict)
num_parents_mating = 5
num_generations = 100  # Zwiększenie liczby pokoleń
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 80
# Lista przechowująca wyniki wartości przedmiotów
best_solutions_values = []

# Licznik udanych prób
successful_attempts = 0
total_time = 0

# Pętla do 10 testów
for i in range(10):
    start = time.time()  # Start pomiaru czasu

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes
                           )

    # Uruchomienie algorytmu
    ga_instance.run()

    # Zakończenie pomiaru czasu
    end = time.time()
    elapsed_time = end - start
    total_time += elapsed_time

    # Podsumowanie
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solutions_values.append((solution, solution_fitness))

    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # ga_instance.plot_fitness()

    # Sprawdzenie czy znaleziono najlepsze rozwiązanie
    if solution_fitness == 1630 or solution_fitness >= 1000:
        successful_attempts += 1

# e) Skuteczność algorytmu
success_rate = successful_attempts / 10 * 100

# f) Średni czas działania algorytmu
average_time = total_time / 10

print("\nSkuteczność algorytmu: {}%".format(success_rate))
print("Średni czas działania algorytmu: {:.6f} sekund".format(average_time))

