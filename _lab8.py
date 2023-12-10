import random
import time

from utils import *


# for all dimensions it is pretty similar
def simulated_annealing_one(func, n, bounds, n_cycles_, m_cycles_, c_, t_, beta_):
    x_init = random.uniform(bounds[0], bounds[1])
    f_init = func(x_init)

    for i in range(n_cycles_):
        for j in range(m_cycles_):
            new_x = x_init + random.uniform(-0.5, 0.5)
            new_x = max(min(new_x, bounds[1]), bounds[0])
            f_new = func(new_x)
            delta_f = abs(f_new - f_init)

            if f_new < f_init or random.uniform(0, 1) < math.exp(-1 * delta_f / (c_ * t_)):
                x_init, f_init = new_x, f_new

        t_ *= beta_

    return x_init, func(x_init)


def simulated_annealing_two(func, n, n_cycles_, m_cycles_, c_, t_, beta_, bounds):
    # x_init = [random.uniform(bounds[0][0], bounds[0][1]), random.uniform(bounds[1][0], bounds[1][1])]
    x_init = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n)]

    x_init.append(func(x_init, n))
    # outer cycle
    for i in range(n_cycles_):
        # inner cycle
        for j in range(m_cycles_):
            # new solution
            new_vector = [el + random.random() - 0.5 for el in x_init[:-1]]
            new_vector = [max(min(el, bounds[i][1]), bounds[i][0]) for i, el in enumerate(new_vector)]
            # fitness of a new solution
            f_new = func(new_vector, n)
            delta_f = abs(f_new - x_init[-1])
            # if new sol is better
            if f_new < x_init[-1]:
                # adding new sol
                x_init = [*new_vector, f_new]
            else:
                # if worse than calc the probability
                r_ = random.uniform(0, 1)
                # Boltzmann distribution
                P = math.exp(-1 * delta_f / (c_ * t_))
                if r_ < P:
                    # if inside the prob, add worse to new sol
                    x_init = [*new_vector, f_new]
        # decreasing temperature
        t_ *= beta_

    return x_init[:-1], func(x_init, n)


def simulated_annealing_n(func, n, n_cycles_, m_cycles_, c_, t_, beta_, bounds):
    x_init = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n)]
    x_init.append(func(x_init, n))
    # outer cycle
    for i in range(n_cycles_):
        # inner cycle
        for j in range(m_cycles_):
            # new solution
            new_vector = [el + random.random() - 0.5 for el in x_init[:-1]]
            new_vector = [max(min(el, bounds[i][1]), bounds[i][0]) for i, el in enumerate(new_vector)]
            # fitness of a new solution
            f_new = func(new_vector, n)
            delta_f = abs(f_new - x_init[-1])
            # if new sol is better
            if f_new < x_init[-1]:
                # adding new sol
                x_init = [*new_vector, f_new]
            else:
                # if worse than calc the probability
                r_ = random.uniform(0, 1)
                # Boltzmann distribution
                P = math.exp(-1 * delta_f / (c_ * t_))
                if r_ < P:
                    # if inside the prob, add worse to new sol
                    x_init = [*new_vector, f_new]
        # decreasing temperature
        t_ *= beta_

    return x_init[:-1], func(x_init, n=n)


def rossenblock(vect, n):
    sum_ = 0
    for i in range(n - 1):
        sum_ += 100 * (vect[i + 1] - vect[i] ** 2) ** 2 + (1 - vect[i]) ** 2
    return sum_


def optimize(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    end = time.time() - start
    print(f'Optimizer: {func.__name__}, Func: {args[0].__name__}, Result: {res}, time: {end}')


t = 100
c = 0.001
beta = 0.99
n_cycles = 500
m_cycles = 400

bounds_rossenblock = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
default_bounds = [(-5, 5), (-5, 5)]
bounds_one = (-5, 5)
optimizers = [simulated_annealing_one, simulated_annealing_two, simulated_annealing_n]
functions = [
    [(func_1, 1, bounds_one), (func_2, 1, bounds_one), (func_3, 1, bounds_one)],
    [(rossenblock, 2, default_bounds), (echli, 2, default_bounds), (cross_in_tray, 2, default_bounds)],
    [(rossenblock, 5, default_bounds * 5), (echli, 2, default_bounds), (cross_in_tray, 2, default_bounds)]
]
# all dims
for i in range(3):
    print(f'{i + 1} dimension{"s" if i + 1 > 1 else ""}:')
    # 3 funcs
    for m in range(len(functions[0])):
        optimize(optimizers[i], functions[i][m][0], bounds=functions[i][m][-1], n=functions[i][m][1],
                 n_cycles_=n_cycles,
                 m_cycles_=m_cycles,
                 c_=c,
                 t_=t,
                 beta_=beta)
