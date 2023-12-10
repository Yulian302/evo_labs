import time
import numpy as np
from utils import *
import random
import itertools


def fractal_structuring_one_dim(function, a, b, n_points, epochs, std=0.1, stopping='IT', tolerance=0.0006):
    # iteration
    t = 0
    # number of offspring solutions
    m_i = 7
    # initial population (normal distribution)
    p_t = np.random.uniform(a, b, n_points)
    # function values of population
    p_t_fitness = np.array([function(sol) for sol in p_t])
    # epochs
    for _ in range(epochs):
        # generating offsprings (7 for each parent solution)
        offsprings = []
        for j in range(len(p_t)):
            for _ in range(m_i):
                # formula: parent + normal distribution (mean=0,std)
                offsprings.append(p_t[j] + np.random.normal(0, std))
        # needed params
        s_l, s_r, m_l, m_r = 0, 0, 0, 0
        # for each offspring
        for o in range(m_i):
            # if offspring_j < p_t_j
            if offsprings[o] < p_t[j]:
                s_l += offsprings[o]
                m_l += 1
            else:
                s_r += offsprings[o]
                m_r += 1
            # other formulas
        x_l_star = s_l / m_l if m_l != 0 else 0
        x_r_star = s_r / m_r if m_r != 0 else 0
        x_i_h = x_l_star if function(x_l_star) > function(x_r_star) else x_r_star
        # adding to offsprings population
        offsprings.append(x_i_h)
        offsprings = np.array(offsprings)
        # p_in population: [p_t + offsprings]
        p_in = np.concatenate((p_t, offsprings), axis=0)
        # sorting in descending order
        p_in = p_in[np.argsort([function(sol) for sol in p_in])[::-1]]
        p_t_prev = p_t.copy()
        p_t = p_in[-n_points:]
        # checking end of algorithm criteria
        if stopping == 'VC':
            epochs += 1
            if np.all(np.abs(p_t[:, np.newaxis] - p_t[np.newaxis, :]) < tolerance):
                break
        elif stopping == 'VF':
            epochs += 1
            if np.abs(np.average([function(sol) for sol in p_t_prev]) - np.average(
                    [function(sol) for sol in p_t])) < tolerance:
                break
            t += 1
        # return best solution (minimum)
    best_solution = p_t[np.argmin(p_t_fitness)]
    return best_solution, function(best_solution)


def fractal_structuring_two_dims(function, bounds, temp_max, n_points, epochs, stopping='IT',
                                 tolerance=0.0006):
    # iteration
    t = 0
    # param
    l = 1
    temperature = temp_max
    # number of offspring solutions
    m_i = 7
    # initial population (normal distribution)
    p_t = np.array(
        [
            [random.uniform(bounds[n_][0], bounds[n_][1]) for n_ in range(2)] for p in range(n_points)
        ]
    )
    # function values of population
    p_t_fitness = np.array([function(sol) for sol in p_t])
    # radius
    r = min(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]) / n_points
    # epochs
    for _ in range(epochs):
        # changing l param
        l /= (t + 1)
        # pv population
        pv = []
        # generating offsprings (7 for each parent solution)
        for j in range(len(p_t)):
            offsprings = []
            for i in range(m_i):
                # creating x1 and x2 by formulas
                x1 = np.random.uniform(p_t[i][0] - 3 * l * r, p_t[i][0] + 3 * l * r)
                x2 = p_t[i][1] + np.random.uniform(-1, 1) * np.sqrt(abs(r ** 2 - (x1 - p_t[i][0]) ** 2))
                offsprings.append([x1, x2])
            offsprings.append(p_t[j])
            pv += offsprings
        # sorting pv in DESC order
        pv = np.array(pv)[np.argsort([function(sol) for sol in pv])[::-1]]
        # 2n best solutions
        best_solutions = pv[-2 * n_points:]
        # pc population
        pc = []
        # generating random pairs and creating new pop from them
        for _ in range(2 * n_points):
            i_, j_ = random.sample(range(2 * n_points), 2)
            pc.append([(best_solutions[i_][0] + best_solutions[j_][0]) / 2,
                       (best_solutions[i_][1] + best_solutions[j_][1]) / 2])
        worst = pv[:2 * n_points]
        f_avg = np.average([function(sol) for sol in best_solutions[:n_points]])
        # pw population
        pw = []
        for w in worst:
            dX = np.random.uniform(
                w[0] - (bounds[0][1] - bounds[0][0]) / n_points,
                w[0] + (bounds[0][1] - bounds[0][0]) / n_points,
            )
            dY = np.random.uniform(
                w[1] - (bounds[1][1] - bounds[1][0]) / n_points,
                w[1] + (bounds[1][1] - bounds[1][0]) / n_points,
            )
            sol = [w[0] + dX, w[1] + dY]
            if function(sol) > f_avg:
                pw.append(sol)
            else:
                r_ = np.random.uniform(0, 1)
                if r_ < np.exp(-np.min([dX, dY]) / temperature):
                    pw.append(sol)
        pw = np.array(pw)
        pc = np.array(pc)
        p_t_old = p_t.copy()
        # updating pt population with pw,pv,pc
        p_t = np.vstack((pw, pv, pc))
        p_t = p_t[np.argsort([function(sol) for sol in p_t])][:n_points]
        if stopping == 'VC':
            epochs += 1
            if np.all(np.abs(p_t[:, np.newaxis, :] - p_t[np.newaxis, :, :]) < tolerance):
                break
        elif stopping == 'VF':
            epochs += 1
            if np.abs(np.average([function(sol) for sol in p_t_old]) - np.average(
                    [function(sol) for sol in p_t])) < tolerance:
                break
        temperature /= 2
        t += 1
    # return best solution (minimum)
    best_solution = p_t[np.argmin(p_t_fitness)]
    return best_solution, function(best_solution)


def fractal_structuring_n_dims(function, n, bounds, temp_max, n_hypersph, epochs, stopping='IT',
                               tolerance=0.0006):
    # iteration
    t = 0
    # param
    l = 1
    temperature = temp_max
    # number of offspring solutions
    m_i = 7
    # initial population (normal distribution)
    p_t = np.array(
        [
            [random.uniform(bounds[n_][0], bounds[n_][1]) for n_ in range(n)] for p in range(n_hypersph)
        ]
    )
    # function values of population
    p_t_fitness = np.array([function(sol) for sol in p_t])
    # radius
    # step 3
    radius = 1 / n_hypersph
    for _ in range(epochs):
        # pv population
        pv = []
        # changing radius
        radius /= (t + 1)
        # generating offsprings (7 for each parent solution)
        # step 4
        for j in range(len(p_t)):
            offsprings = []
            # generating offsprings
            for i in range(m_i):
                # each offspring is vector where random axis is changed with one formula, others with different
                k = random.choice(range(n))
                random_vector = p_t[j].copy()
                for n_ in range(n):
                    if n_ != k:
                        random_vector[n_] = random.uniform(p_t[j][n_] - radius, p_t[j][n_] + radius)
                sum__ = sum([(random_vector[n_] - p_t[j][n_]) ** 2 for n_ in range(n) if n_ != k])
                random_vector[k] = p_t[j][k] + random.uniform(-1, 1) * (radius ** 2 - sum__)
                offsprings.append(random_vector)
            offsprings.append(p_t[j])
            pv += offsprings
        # list to numpy array
        pv = np.array(pv)
        # step 5
        # population pc (exploring new solutions)
        pc = []
        for _ in range(len(pv)):
            i_, j_ = random.sample(range(len(pv)), 2)
            axis = random.randint(0, n)
            random_num = np.random.choice([-1, 1])
            pc.append(
                list([np.where(np.arange(n) == axis, (pv[i_] + pv[j_]) / 2,
                               (radius == -1) * pv[i_] + (radius == 1) * pv[j_])])
            )
        pc = np.array(pc).reshape(-1, n)
        # step 6
        # expliring new solutions
        pv = pv[np.argsort([function(sol) for sol in pv])[::-1]]
        pv_f_avg = np.average([function(sol) for sol in pv])
        # 2n worst solutions
        # creating random pairs
        all_pairs = list(itertools.combinations(pv, 2))
        np.random.shuffle(all_pairs)
        worst_pairs = np.array(all_pairs[:2 * n])
        # for each worst pair
        # searching better solutions from worst
        pw = []
        for w in worst_pairs.reshape(-1, n):
            rnd_axis_idx = random.choice(range(n))
            deltas_ = np.zeros(n)
            d_axis = np.random.uniform(
                w[rnd_axis_idx] - ((bounds[rnd_axis_idx][1] - bounds[rnd_axis_idx][0]) / n),
                w[rnd_axis_idx] + ((bounds[rnd_axis_idx][1] - bounds[rnd_axis_idx][0]) / n)
            )
            deltas_[rnd_axis_idx] = d_axis
            w[rnd_axis_idx] += d_axis
            if function(w) > pv_f_avg:
                pw.append(w)
            else:
                rnd = np.random.uniform(0, 1)
                if rnd <= np.exp(-np.min(deltas_) / temperature):
                    pw.append(w)
        pw = np.array(pw)
        p_t_old = p_t.copy()
        p_t = np.vstack((pw, pv, pc))
        p_t = p_t[np.argsort([function(sol) for sol in p_t])][:n_hypersph]
        if stopping == 'VC':
            epochs += 1
            if np.all(np.abs(p_t[:, np.newaxis, :] - p_t[np.newaxis, :, :]) < tolerance):
                break
        elif stopping == 'VF':
            epochs += 1
            if np.abs(np.average([function(sol) for sol in p_t_old]) - np.average(
                    [function(sol) for sol in p_t])) < tolerance:
                break
        temperature /= 2
        t += 1
    best_solution = p_t[np.argmin([function(elem) for elem in p_t])]
    return np.round(best_solution, 5), np.round(function(best_solution), 5)


# bounds for cross_n_tray
cross_n_tray_bounds = [(-10, 10), (-10, 10)]
rosenbrock_bounds = [(-10, 10), (-10, 10)]
echli_bounds = [(-5, 5), (-5, 5)]
features = ['IT', 'VC', 'VF']
columns = ['result', 'time']
values = [10, 20, 30]

# for two dims
# bounds = [rosenbrock_bounds, cross_n_tray_bounds, echli_bounds]
# functions = [rosenbrock, cross_in_tray, echli]

# for one dim
# bounds for one dim
a, b = -5, 5
# functions = [func_1, func_2, func_3]

# for n dims
bounds = [[(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)],
          cross_n_tray_bounds, echli_bounds]
functions = [(rosenbrock, 9), (cross_in_tray, 2), (echli, 2)]

# generating dataframe (table) of results
for m in range(len(functions)):
    solutions = []
    time_ = []
    for i in range(len(features)):
        start = time.time()
        # fractal
        solutions.append(
            fractal_structuring_n_dims(functions[m][0], functions[m][1], bounds[m], 10000, n_hypersph=30,
                                       stopping=features[m],
                                       epochs=100,
                                       tolerance=0.0006))
        time_.append(time.time() - start)

    generate_df(features, columns, [(solutions[i_], time_[i_]) for i_ in range(len(features))])
