import math
import pandas as pd
from tabulate import tabulate
import numpy as np


def rosenbrock(vect, n=2):
    sum = 0
    for n_ in range(n - 1):
        sum += 100 * (vect[n_ + 1] - vect[n_] ** 2) ** 2 + (1 - vect[n_]) ** 2
    return sum


def rastrigina_func(vect, n=7):
    a = 10
    sum_ = 0
    for i in range(n):
        sum_ += vect[i] ** 2 - a * np.cos(2 * np.pi * vect[i])
    return a * n + sum_


def cross_in_tray(vect, n=2):
    return (
        -0.0001
        * (
            abs(
                math.sin(vect[0])
                * math.sin(vect[1])
                * math.exp(
                    abs(100 - (math.sqrt(vect[0] ** 2 + vect[1] ** 2) / math.pi))
                )
                + 1
            )
        )
        ** 0.1
    )


def echli(vect, n=2):
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (vect[0] ** 2 + vect[1] ** 2)))
        - math.exp(
            0.5 * (math.cos(2 * math.pi * vect[0]) + math.cos(2 * math.pi * vect[1]))
        )
        + math.e
        + 20
    )


# for creating dataframe(table)
def generate_df(features, columns, data):
    df = pd.DataFrame(index=features, columns=columns, data=data)
    print(tabulate(df, headers="keys", tablefmt="psql"))


# one dim funcs
def func_1(x):
    return x**2


def func_2(x):
    return x**2 + 45


def func_3(x):
    return 5 * x**2
