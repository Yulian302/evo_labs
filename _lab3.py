import random
from utils import *
import time


def symbiotic_optimization(function, population_size, bounds, eco_size, max_epochs=100, max_fit_eval=100, stopping='IT',
                           bfs=None,
                           tolerance=0.0005):
    # ecosystem (==population)
    eco_system = [(random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])) for _ in
                  range(population_size)]
    # organisms (< population)
    organisms = random.choices(eco_system, k=eco_size)
    t = 0
    i = 0
    organisms_fitness = []
    num_fit_eval = 0
    while True:
        previous_population = list(organisms)
        fitness = [function(p, n=2) for p in organisms]
        best_org_idx = fitness.index(min(fitness))
        best_organism = organisms[best_org_idx]
        # mutualism
        x_i = organisms[i]
        x_j = random.choice(organisms[:i] + organisms[i + 1:])
        x_j_idx = organisms.index(x_j)
        mutual_vector = ((x_i[0] + x_j[0]) / 2, (x_i[1] + x_j[1]) / 2)
        if not bfs:
            bf1 = random.choice([1, 2])
            bf2 = random.choice([1, 2])
        else:
            bf1 = bfs[0]
            bf2 = bfs[1]

        x_i_new = (x_i[0] + random.random() * (best_organism[0] - mutual_vector[0] * bf1),
                   x_i[1] + random.random() * (best_organism[1] - mutual_vector[1] * bf1))
        x_j_new = (x_j[0] + random.random() * (best_organism[0] - mutual_vector[0] * bf2),
                   x_j[1] + random.random() * (best_organism[1] - mutual_vector[1] * bf2))
        organisms_fitness.append(function(x_i_new))
        organisms_fitness.append(function(x_j_new))
        x_i_new_fit = function(x_i_new)
        num_fit_eval += 2
        # check if new organisms are better (if so, then substitute)
        if x_i_new_fit < function(x_i_new) and x_i_new_fit < function(x_j_new):
            organisms[i] = x_i_new
            organisms[x_j_idx] = x_j_new
        # commensalism
        x_j = random.choice(organisms[:i] + organisms[i + 1:])
        x_i_new = (x_i[0] + random.uniform(-1, 1) * (best_organism[0] - x_j[0]),
                   x_i[1] + random.uniform(-1, 1) * (best_organism[1] - x_j[1]))
        organisms_fitness.append(function(x_i_new))
        num_fit_eval += 1
        if function(x_i_new) < function(x_i):
            organisms[i] = x_i_new
        else:
            x_i_new = None
        # parasitism
        x_j = random.choice(organisms[:i] + organisms[i + 1:])
        x_j_idx = organisms.index(x_j)
        r = random.choice([0, 1])
        if r == 0:
            parasite_vector = (x_i[0] + random.uniform(-1, 1), x_i[1])
        else:
            parasite_vector = (x_i[0], x_i[1] + random.uniform(-1, 1))

        parasite_fitness = function(parasite_vector)
        organisms_fitness.append(parasite_fitness)
        num_fit_eval += 1

        if function(parasite_vector) < function(x_j):
            organisms[x_j_idx] = parasite_vector

        # VC (distance between population individuals < tolerance)
        if stopping == 'VC':
            xs = [x for x, y in organisms]
            ys = [y for x, y in organisms]
            if all(abs(a - b) < tolerance for a, b in zip(xs, xs[1:])) and all(
                    abs(a - b) < tolerance for a, b in zip(ys, ys[1:])):
                best = min(organisms, key=lambda organism: function(organism))
                return best, function(best)

        # VF (mean fitness between neighbour populations < tolerance)
        elif stopping == 'VF':
            current_population = organisms
            previous_average = sum(function(p) for p in previous_population) / len(previous_population)
            current_average = sum(function(p) for p in current_population) / len(current_population)

            if abs(current_average - previous_average) < tolerance:
                best = min(organisms, key=lambda organism: function(organism))
                return best, function(best)

        if i == eco_size - 1:
            if stopping == 'IT' and t > max_epochs:  # or num_fit_eval > max_fit_eval:
                best = min(organisms, key=lambda organism: function(organism))
                return best, function(best)
            else:
                t += 1
                i = 0
        i += 1


# funcs data
bounds = [(-5.12, 5.12), (-5, 5), (-5, 5)]
functions = [(rosenbrock, 2), (cross_in_tray, 2), (echli, 2)]
features = ['IT', 'VC', 'VF']
columns = ['result', 'time']
feature_values_1 = [(1, random.choice([1, 2])), (2, random.choice([1, 2]))]
feature_values_2 = [(random.choice([1, 2]), 1), (random.choice([1, 2]), 2)]
# generating dataframe (table) of results
for m in range(len(functions)):
    solutions = []
    time_ = []
    for i in range(len(features)):
        start = time.time()
        solutions.append(
            symbiotic_optimization(functions[m][0], population_size=20, eco_size=5, max_epochs=200,
                                   max_fit_eval=int(1e20),
                                   stopping='IT', tolerance=0.0001, bounds=bounds[m], bfs=None)
        )
        time_.append(time.time() - start)

    generate_df(features, columns, [(solutions[i_], time_[i_]) for i_ in range(len(features))])
