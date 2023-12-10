import time
import random
from utils import *


def calc_neighbour_source(src, rnd_src, a):
    xv = src[0] + (src[0] - rnd_src[0]) * random.uniform(-a, a)
    yv = src[1] + (src[1] - rnd_src[1]) * random.uniform(-a, a)
    return (xv, yv)


def bees_optimization(func, m, bounds, a=0.1, epochs=100, tolerance=0.00001, stopping='IT'):
    # bees-scouts
    bees_scouts = [
        (bounds[0] + random.random() * (bounds[1] - bounds[0]), bounds[0] + random.random() * (bounds[1] - bounds[0]))
        for _ in
        range(m)]

    for _ in range(epochs):
        previous_population = list(bees_scouts)
        # bees-workers
        # bees workers search for new food sources
        for i in range(len(bees_scouts)):
            # searching for new sources for random bee scout
            k = random.choice(range(len(bees_scouts)))
            x_k = bees_scouts[k]
            # neighbour source
            v_ij = calc_neighbour_source(bees_scouts[i], x_k, a)
            # its fitness
            v_i_fit = func(v_ij)
            # if found source better than current bee scout
            x_i_fit = func(bees_scouts[i])
            chosen = v_ij if v_i_fit < x_i_fit else bees_scouts[i]
            # write better to bees_scout_i
            bees_scouts[i] = chosen

        # onlooker-bees
        # onlooker bees get info about possible sources from bees workers and go to find it
        total_fitness = sum(func(bees_scout) for bees_scout in bees_scouts)
        onlooker_bees = []

        for i in range(len(bees_scouts)):
            p = func(bees_scouts[i]) / total_fitness
            if random.random() < p:
                onlooker_bees.append(bees_scouts[i])

        for i in range(len(onlooker_bees)):
            k = random.choice(range(len(bees_scouts)))
            x_k = bees_scouts[k]
            # a - param: how far to search for sources of food
            v_i = calc_neighbour_source(onlooker_bees[i], x_k, a)
            x_i_fit = func(onlooker_bees[i])
            v_i_fit = func(v_i)
            chosen_new = v_i if x_i_fit < v_i_fit else onlooker_bees[i]
            onlooker_bees[i] = chosen_new
        if stopping == 'VC':
            epochs += 1
            try:
                delta_f = abs(func(chosen) - func(chosen_new))
            except:
                delta_f = 99
            if delta_f < tolerance:
                break
        elif stopping == 'VF':
            current_population = bees_scouts
            previous_average = sum(func(p) for p in previous_population) / len(previous_population)
            current_average = sum(func(p) for p in current_population) / len(current_population)
            epochs += 1
            if abs(current_average - previous_average) < tolerance:
                break

    best_solution = min(bees_scouts, key=lambda x: func(x))
    best_fitness = func(best_solution)
    return best_solution, best_fitness


# two dims
n = 2
M = 40
functions: list[tuple] = [(rosenbrock, [-10, 10]), (echli, [-5, 5]), (cross_in_tray, [-10, 10])]
features = ['A1', 'A2', 'A3']
values = [1, 0.7, 0.01]
columns = ['X', 'Y', 'Z', 'Time']
for m in range(len(functions)):
    func_res = []
    times = []
    for i in range(len(features)):
        start = time.time()
        func_res.append(
            bees_optimization(functions[m][0], M, functions[m][1], a=values[m], epochs=100, stopping='IT',
                              tolerance=0.0000001))
        times.append(time.time() - start)
    generate_df(features, columns,
                data=[(func_res[q][0][0], func_res[q][0][1], func_res[q][1], times[q]) for q in
                      range(len(features))])
