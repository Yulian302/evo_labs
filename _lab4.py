import collections
import random
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import copy

# reading distance and city names from csv
distance_matrix = []
cities_df = pd.read_csv("lab4.csv", sep=";")
for c in range(1, 26):
    distance_matrix.append(cities_df[str(c)].values)


class AntSystem(object):
    def __init__(self):
        pass

    # methods for generating initial pheromones
    @staticmethod
    def as_initial_pheromone():
        return [[random.random() for _ in range(n_cities)] for _ in range(n_cities)]

    @staticmethod
    def mmas_initial_pheromone():
        return [[max_tau for _ in range(n_cities)] for _ in range(n_cities)]

    # performing local search using 2-opt heuristic
    def local_search(self, ant_paths):
        temp = []
        for i in range(len(ant_paths)):
            tour, path = self.two_opt(ant_paths[i][0])
            temp.append((tour, path))
        return temp

    # implementing local search using 2-opt heuristic
    def two_opt(self, tour):
        n_cities = len(tour)
        best = tour
        better = True

        while better:
            better = False
            for i in range(1, n_cities - 1):
                for j in range(i + 1, n_cities):
                    if j - i == 1:
                        continue
                    new = tour[:i] + tour[i:j][::-1] + tour[j:]
                    current_distance = self.calc_total_dist(tour)
                    new_distance = self.calc_total_dist(new)
                    if new_distance < current_distance:
                        tour = new
                        better = True

            if better:
                best = tour

        return best, self.calc_total_dist(best)

    # calculate probabilities for selecting the next city
    def calc_probs(self, current, visited):
        probs = []
        total_prob = 0
        for city in range(n_cities):
            if city not in visited:
                pheromone = pheromone_matrix[current][city]
                dist = distance_matrix[current][city]
                probability = math.pow(pheromone, alpha) * math.pow(1 / dist, beta)
                probs.append((city, probability))
                total_prob += probability
        normalized_probs = [
            (city, probability / total_prob) for city, probability in probs
        ]
        return normalized_probs

    # ant system algorithm (AS)
    def as_algo(self, num_ants, num_iterations, best_dist=False, ls_flag=False):
        avg = []
        start = time.perf_counter()
        global pheromone_matrix, paths, total_distance, visited
        best_path = None
        best_dist = float("inf")

        for _ in range(num_iterations):
            paths = []
            for ant in range(num_ants):
                current_city = 6
                visited = [current_city]
                total_distance = 0

                while len(visited) < n_cities:
                    probs = self.calc_probs(current_city, visited)

                    selected_city = random.choices(
                        probs, weights=[p for _, p in probs], k=1
                    )[0][0]
                    visited.append(selected_city)
                    total_distance += distance_matrix[current_city][selected_city]
                    current_city = selected_city
            total_distance += distance_matrix[visited[-1]][visited[0]]
            visited.append(6)
            paths.append((visited, total_distance))

        if ls_flag:
            paths = self.local_search(paths)
        if best_dist > min(paths, key=lambda x: x[1])[1]:
            best_path, best_dist = min(paths, key=lambda x: x[1])

        iter_best_path, iter_best_dist = min(paths, key=lambda x: x[1])
        avg.append(iter_best_dist)
        # Updating pheromone level
        pheromone_matrix = [
            [(1 - evaporate_rate) * pheromone for pheromone in row]
            for row in pheromone_matrix
        ]
        if best_dist:
            for path, distance in paths:
                for i in range(len(path) - 1):
                    if path != best_path and distance != best_dist:
                        pheromone_matrix[path[i]][path[i + 1]] += 1 / distance
                    else:
                        pheromone_matrix[path[i]][path[i + 1]] += (1 / distance) + E * (
                            1 / distance
                        )
                if path != best_path and distance != best_dist:
                    pheromone_matrix[path[-1]][path[0]] += 1 / distance
                else:
                    pheromone_matrix[path[-1]][path[0]] += (1 / distance) + E * (
                        1 / distance
                    )
        else:
            for path, distance in paths:
                for i in range(len(path) - 1):
                    pheromone_matrix[path[i]][path[i + 1]] += 1 / distance
                pheromone_matrix[path[-1]][path[0]] += 1 / distance
        print("Best distance:", best_distance)
        print("Best path:", [cities_df["Unnamed: 1"][i] for i in best_path])
        return best_path, best_dist, time.perf_counter() - start, avg

    # calculating total distance
    def calc_total_dist(self, tour):
        total_dist = 0
        n_cities = len(tour)
        for i in range(n_cities):
            total_dist += distance_matrix[tour[i - 1]][tour[i]]
        return total_dist

    # max min ant system algorithm (MMAS)
    def mmas_algo(self, num_ants, num_iterations, ls_flag=False):
        avg = []
        start = time.perf_counter()
        global pheromone_matrix
        best_path = None
        best_distance = float("inf")

        for _ in range(num_iterations):
            ant_paths = []
            for ant in range(num_ants):
                current_city = 6
                visited_cities = [current_city]
                total_distance = 0

                while len(visited_cities) < n_cities:
                    probs = self.calc_probs(current_city, visited_cities)
                    selected_city = random.choices(
                        probs, weights=[p for _, p in probs], k=1
                    )[0][0]
                    visited_cities.append(selected_city)
                    total_distance += distance_matrix[current_city][selected_city]
                    current_city = selected_city

                total_distance += distance_matrix[visited_cities[-1]][visited_cities[0]]
                visited_cities.append(6)
                ant_paths.append((visited_cities, total_distance))

            if ls_flag:
                ant_paths = self.local_search(ant_paths)
            if best_distance > min(ant_paths, key=lambda x: x[1])[1]:
                best_path, best_distance = min(ant_paths, key=lambda x: x[1])

            iter_best_path, iter_best_dist = min(ant_paths, key=lambda x: x[1])
            avg.append(iter_best_dist)

            pheromone_matrix = [
                [(1 - evaporate_rate) * pheromone for pheromone in row]
                for row in pheromone_matrix
            ]
            for i in range(len(iter_best_path) - 1):
                pheromone_matrix[iter_best_path[i]][iter_best_path[i + 1]] += np.clip(
                    (1 / iter_best_dist), tau_min, max_tau
                )
            pheromone_matrix[iter_best_path[-1]][iter_best_path[0]] += np.clip(
                (1 / iter_best_dist), tau_min, max_tau
            )
        print("Best distance:", best_distance)
        print("Best path:", [cities_df["Unnamed: 1"][i] for i in best_path])
        return best_path, best_distance, time.perf_counter() - start, avg

    # ant colony system algorithm (ACS)
    def acs_algo(self, n_ants, n_iterations, ls_flag=False):
        avg = []
        start = time.perf_counter()
        global pheromone_matrix, paths, visited, current_city
        best_path_ = None
        best_distance_ = float("inf")

        for _ in range(n_iterations):
            paths = []
            for ant in range(n_ants):
                current_city = 6  # kyiv
                visited = [current_city]
            total_dist_ = 0

            while len(visited) < n_cities:
                probs = self.calc_probs(current_city, visited)
                if np.random.uniform() <= q0:
                    selected_city = max(probs, key=lambda x: x[1])[0]
                else:
                    selected_city = random.choices(
                        probs, weights=[p for _, p in probs], k=1
                    )[0][0]
                pheromone_matrix[current_city][selected_city] = (
                    1 - eta
                ) * pheromone_matrix[current_city][
                    selected_city
                ] + eta * init_pheromone_matrix[
                    current_city
                ][
                    selected_city
                ]
                visited.append(selected_city)
                total_dist_ += distance_matrix[current_city][selected_city]
                current_city = selected_city

            total_dist_ += distance_matrix[visited[-1]][visited[0]]
            pheromone_matrix[visited[-1]][visited[0]] = (1 - eta) * pheromone_matrix[
                visited[-1]
            ][visited[0]] + eta * init_pheromone_matrix[visited[-1]][visited[0]]
            visited.append(6)
            paths.append((visited, total_dist_))

        if ls_flag:
            paths = self.local_search(paths)
        if best_distance_ > min(paths, key=lambda x: x[1])[1]:
            best_path_, best_distance_ = min(paths, key=lambda x: x[1])

        iter_best_path, iter_best_dist = min(paths, key=lambda x: x[1])
        avg.append(iter_best_dist)

        pheromone_matrix = [
            [(1 - evaporate_rate) * pheromone for pheromone in row]
            for row in pheromone_matrix
        ]
        for i in range(len(best_path_) - 1):
            pheromone_matrix[best_path_[i]][best_path_[i + 1]] += evaporate_rate * (
                1 / best_distance_
            )
        pheromone_matrix[best_path_[-1]][best_path_[0]] += evaporate_rate * (
            1 / best_distance_
        )
        print("Best distance:", best_distance_)
        print("Best path:", [cities_df["Unnamed: 1"][i] for i in best_path_])
        return best_path_, best_distance_, time.perf_counter() - start, avg


evaporate_rate = 0
best_distance, exec_time, avg, best_path = 0, 0, 0, None
E = 0
init_pheromone_matrix = 0
pheromone_matrix = 0
tau_min, max_tau = 0, 0
q0 = 0.0
eta = 0.0


# training methods for different strategies
def as_train():
    global evaporate_rate, pheromone_matrix, best_path, best_distance, exec_time, avg
    evaporate_rate = 0.5  # MMAS=0.2, ACS = 0.1
    E = 10
    pheromone_matrix = ant_system.as_initial_pheromone()
    best_path, best_distance, exec_time, avg = ant_system.as_algo(
        n_ants, n_iterations, best_dist=False, ls_flag=False
    )


def mmas_train():
    global evaporate_rate, pheromone_matrix, tau_min, max_tau, best_path, best_distance, exec_time, avg
    evaporate_rate = 0.2  # MMAS=0.2, ACS = 0.1
    max_tau = 1
    a = 100
    tau_min = max_tau / a
    pheromone_matrix = ant_system.mmas_initial_pheromone()
    best_path, best_distance, exec_time, avg = ant_system.mmas_algo(
        n_ants, n_iterations, ls_flag=False
    )


def acs_train():
    global evaporate_rate, pheromone_matrix, q0, eta, init_pheromone_matrix, best_path, best_distance, exec_time, avg
    evaporate_rate = 0.2  # MMAS=0.2, ACS = 0.1
    q0 = 0.8
    eta = 0.1
    pheromone_matrix = ant_system.as_initial_pheromone()
    init_pheromone_matrix = copy.deepcopy(pheromone_matrix)
    best_path, best_distance, exec_time, avg = ant_system.acs_algo(
        n_ants, n_iterations, ls_flag=True
    )


# strategy for ant system
strategy = "ACS"
ant_system = AntSystem()
n_iterations = 45
n_ants = 20
alpha = 2
# values between 2 and 5
beta = 4
n_attempts = 50
solutions = []
n_cities = len(distance_matrix)
ant_strategies = {"AS": as_train, "MMAS": mmas_train, "ACS": acs_train}
try:
    for _ in range(n_attempts):
        ant_strategies[strategy]()
        solutions.append((best_distance, best_path))
except KeyError as e:
    raise ValueError("Undefined strategy: {}".format(e.args[0]))


print("Time: ", exec_time)
best_overall = min(solutions, key=lambda s: s[0])
print(f"Best solution overall: {best_overall}")
