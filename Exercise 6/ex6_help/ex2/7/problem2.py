# Arttu Hietalahti 262981

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def fun1(x): return x ** 2 * np.exp(-2 * x)


def fun2(x): return 1 / x * np.sin(x)


def fun3(x): return np.exp(np.sin(x ** 3))


def fun4(x, y): return x * np.exp(-1 * np.sqrt(x ** 2 + y ** 2))


def fun5(x, y, z, xa, ya, za, xb, yb, zb):
    # takes x, y, and z position as input
    # also needs x, y and z values for points A and B to calculate vector norm
    distance_from_A = np.sqrt((x - xa) ** 2 + (y - ya) ** 2 + (z - za) ** 2)
    distance_from_B = np.sqrt((x - xb) ** 2 + (y - yb) ** 2 + (z - zb) ** 2)

    return np.exp(-2 * distance_from_A) / (np.pi * distance_from_B)  # function from problem 2C


def calculate_integrals():
    # problem 2A1
    x1 = np.arange(0, 10000, 0.01)
    int1 = simps(fun1(x1), x1)
    correct_int = 0.25
    print("___________________________PROBLEM 2A1__________________________")
    print("Evaluated integral of r^2*exp(-2*r) through [0, Inf]")
    print("Simpson integral value (step size 0.01): " + str(int1))
    print("Correct analytical value: " + str(correct_int))
    print("Relative error: " + str(100 * (int1 / correct_int - 1)) + " %\n")

    # problem 2A2
    x2 = np.arange(1e-15, 1, 0.01)  # start from epsilon to avoid division by zero
    int2 = simps(fun2(x2), x2)
    correct_int = 0.9460830703672

    print("___________________________PROBLEM 2A2__________________________")
    print("Evaluated integral of sin(x)/x through [0, 1]")
    print("Note: had to start from 1e-15 to avoid division by zero")
    print("Simpson integral value (step size 0.01): " + str(int2))
    print("Correct analytical value: " + str(correct_int))
    print("Relative error: " + str(100 * (int2 / correct_int - 1)) + " %\n")

    # problem 2A3
    x3 = np.arange(0, 5, 0.01)
    int3 = simps(fun3(x3), x3)
    correct_int = 6.64727207995

    print("___________________________PROBLEM 2A3__________________________")
    print("Evaluated integral of exp(sin(x^3)) through [0, 5]")
    print("Simpson integral value (step size 0.01): " + str(int3))
    print("Correct analytical value: " + str(correct_int))
    print("Relative error: " + str(100 * (int3 / correct_int - 1)) + " %\n")

    # problem 2B
    gridpoints = 100
    x = np.linspace(0, 2, gridpoints)
    y = np.linspace(-2, 2, gridpoints)

    [X, Y] = np.meshgrid(x, y)  # use meshgrid for 2D evaluation of function
    F = fun4(X, Y)  # function values at grid points
    int_dx = simps(F, x, axis=0)  # simpson integral over x dimension
    int4 = simps(int_dx, y)  # simpson integral over y dimension
    correct_int = 1.573477135767718  # from MATLAB numerical integration (AbsTol 1e-18)

    print("___________________________PROBLEM 2B__________________________")
    print("Evaluated integral of x*exp(-sqrt(x^2+y^2) through x=[0, 2] and y = [-2, 2]")
    print("Simpson integral value: " + str(int4))
    print("Correct analytical value: " + str(correct_int))
    print("Relative error: " + str(100 * (int4 / correct_int - 1)) + " %\n")

    # problem 2C
    [x_a, y_a, z_a] = [2, 2, 2]  # point A
    [x_b, y_b, z_b] = [4, 4, 4]  # point B
    gridpoints = 100

    x = np.linspace(x_a, x_b, gridpoints)
    y = np.linspace(y_a, y_b, gridpoints)
    z = np.linspace(z_a, z_b, gridpoints)
    [X, Y, Z] = np.meshgrid(x, y, z)  # use meshgrid for 2D evaluation of function
    F = fun5(X, Y, Z, xa=x_a, ya=y_a, za=z_a, xb=x_b, yb=y_b, zb=z_b)  # function values at grid points
    F[F == np.inf] = 0  # remove infs
    F[F == np.nan] = 0  # remove nans

    int_dx = simps(F, x, axis=0)  # first integrate through x dimension
    int_dxy = simps(int_dx, y, axis=0)  # then integrate through y dimension
    int5 = simps(int_dxy, z)  # third integration through z dimension

    # analytical result from exercise sheet
    R = np.sqrt((x_a-x_b)**2 + (y_a-y_b)**2 + (z_a-z_b)**2)
    analytical_int = (1 - (1 + R) * np.exp(-2 * R)) / R

    print("___________________________PROBLEM 2C__________________________")
    print("Evaluated integral from problem 2C from r_a = [2, 2, 2] to r_b = [4, 4, 4]")
    print("Simpson integral value: " + str(int5))
    print("Correct analytical value: " + str(analytical_int))
    print("Relative error: " + str(100 * (int5 / analytical_int - 1)) + " %\n")
    # three-dimensional simpson integral gives wrong answer (about 1 order of magnitude smaller)
    # probably due to the infinity tendency of the function at r = rb
    # i don't really know how to fix this
    # mathematics is hard


def main():
    calculate_integrals()


if __name__ == "__main__":
    main()
