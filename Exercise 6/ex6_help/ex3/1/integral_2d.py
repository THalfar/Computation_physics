import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

""" FYS-4096 Computational Physics: Exercise 3 """
""" Author: Santeri Saariokari """

def f(x, y):
    """
    Test function for 2D integration
    """
    return (x+y) * np.exp(-np.sqrt(x**2 + y**2))

def test_simps():
    """
    Calculates a 2D integral using simps
    """
    correct_value = 1.57348 # Wolfram alpha
    grid_sizes = np.linspace(10, 100, 90, dtype=int)
    integrals = []
    for points in grid_sizes:
        x = np.linspace(0, 2, points)
        y = np.linspace(-2, 2, points)
        X,Y = np.meshgrid(x,y)
        Z = f(X,Y)
        # integration over x
        I_x = simps(Z, dx=x[1] - x[0])
        # integration over y
        I_xy = simps(I_x, dx=y[1] - y[0])
        integrals.append(I_xy)
        print("Numerical result: %.5f, Grid size: %.5f, Error: %.10f" % (I_xy, points, abs(I_xy - correct_value)))
    plot_convergence(grid_sizes, integrals)

def plot_convergence(grid_sizes, integrals):
    """
    Draws "integrals" as function of "grid_sizes"
    """
    fig = plt.figure()
    plt.plot(grid_sizes, integrals)
    plt.xlabel("Points")
    plt.ylabel("Integral")
    plt.show()

def main():
    test_simps()

if __name__ == "__main__":
    main()