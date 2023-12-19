import random
from utils import *
import time
import numpy as np


def check_if_in_range(p, bounds):
    is_within_range_ = True
    for i in range(len(p)):
        if not (bounds[i][0] <= p[i] <= bounds[i][1]):
            is_within_range_ = False
    return is_within_range_


def deformed_stars(
    function,
    bounds,
    n,
    n_triangles,
    vertices=3,
    k=3,
    alpha=30,
    epochs=50,
    stopping="IT",
    tolerance=0.009,
):
    # triangle, new triangle
    tr_, new_tr = None, None
    # init triangles (each triangle = three points)
    triangles_ = np.asarray(
        [
            np.array(
                [
                    [random.uniform(bounds[n_][0], bounds[n_][1]) for n_ in range(n)]
                    for _ in range(vertices)
                ]
            )
            for _ in range(n_triangles)
        ]
    )
    epoch = 0
    while epoch < epochs:
        new_triangles_ = []
        # for each triangle...
        for tr_, t in zip(triangles_, range(n_triangles)):
            # min vertice (вершина)
            min_vert = np.argmin([function(p, n=n) for p in tr_])
            new_vert = (1 / (k - 1)) * (k * min_vert - np.mean(tr_, axis=0))
            # creating new triangle
            new_tr = np.array(
                [
                    new_vert,
                    (1 / k) * (k - 1) * tr_[1] + new_vert,
                    (1 / k) * (k - 1) * tr_[2] + new_vert,
                ]
            )
            # rotating triangle (searching for better solution)
            i, k_, l_ = random.sample(range(vertices), 3)
            rotated_tr = tr_.copy()
            rotated_tr[k_] = tr_[k_] * np.cos(alpha) - tr_[l_] * np.sin(alpha)
            rotated_tr[l_] = tr_[k_] * np.sin(alpha) + tr_[l_] * np.cos(alpha)
            # shrinking triangle (searching for better solution)
            shrinked_tr = tr_.copy()
            for i in range(vertices):
                if i == min_vert:
                    continue
                shrinked_tr[i] = (k * tr_[min_vert] + shrinked_tr[i]) / (1 + k)
            new_triangles_.extend([new_tr, rotated_tr, shrinked_tr])

        # new triangles
        new_triangles_ = np.array(
            [
                t
                for t in new_triangles_
                if all(check_if_in_range(vertex, bounds) for vertex in t)
            ]
        )
        # sorted fitness (func value for each triangle)
        fitness_sorted = np.argsort(
            np.array([function(np.mean(p, axis=0), n=n) for p in new_triangles_])
        )
        # n best triangles
        triangles_ = new_triangles_[fitness_sorted][:n_triangles]
        # stopping criteria
        if stopping == "VC":
            end = False
            for tr_ in triangles_:
                if np.all(
                    np.linalg.norm(
                        tr_[:, np.newaxis, :] - tr_[:, :, np.newaxis], axis=0
                    )
                    < tolerance
                ):
                    end = True
            if end:
                break
        elif stopping == "VF":
            if (
                np.abs(
                    np.mean([function(v_) for v_ in tr_])
                    - np.mean([function(v_) for v_ in new_tr])
                )
                < tolerance
            ):
                break
        else:
            epoch += 1
    # flatten triangles (so that we have just points)
    tr_flatten = triangles_.flatten().reshape(-1, n)
    # fitness of each point
    fitness_ = [function(p) for p in tr_flatten]
    # min fitness point index
    min_fit_idx = np.argmin(fitness_)
    # best solution (best point and it's func value)
    best_solution = tr_flatten[min_fit_idx]
    best_fitness = fitness_[min_fit_idx]
    # rounding for better view
    return np.round(best_solution, 5), round(best_fitness, 3)


funcs = [[rosenbrock, 4], [cross_in_tray, 2], [echli, 2], [rastrigina_func, 7]]
features = ["IT", "VC", "VF"]
bounds_ = [
    [
        (-5.12, 5.12),
        (-5.12, 5.12),
        (-5.12, 5.12),
        (-5.12, 5.12),
    ],
    [(-5, 5), (-5, 5)],
    [(-5, 5), (-5, 5)],
    [(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)],
]
tols_ = [0.1, 0.2, 0.009]
vals = [10, 20, 30]
columns = ["Result", "Time"]


for m in range(len(funcs)):
    solutions = []
    time_ = []
    for i in range(len(features)):
        start = time.time()
        # deformed stars method
        solutions.append(
            deformed_stars(
                funcs[m][0],
                bounds_[m],
                funcs[m][1],
                n_triangles=10,
                vertices=3,
                k=3,
                tolerance=0.009,
                epochs=50,
                stopping="IT",
            )
        )
        time_.append(time.time() - start)

    generate_df(
        features, columns, [(solutions[i_], time_[i_]) for i_ in range(len(features))]
    )
