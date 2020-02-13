import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

from spline_class import spline

def f(x, y):
    """
    Test function for 2D interpolation
    """
    return (x+y) * np.exp(-np.sqrt(x**2 + y**2))

def interpolate():
    """
    Generates "experimental data" and estimates the values using interpolation
    """
    x_grid = np.linspace(-2, 2, 30)
    y_grid = np.linspace(-2, 2, 30)
    # grid points used for spline
    X,Y = np.meshgrid(x_grid, y_grid)
    # spline object from X,Y
    spline_2d = spline(x=x_grid, y=y_grid, f=f(X,Y), dims=2)

    # x must be chosen so that y does not go over 2
    x = np.linspace(0, 2/np.sqrt(1.75), 100)
    y = np.sqrt(1.75) * x
    spline_eval = []
    for point in range(len(x)):
        # go through containers x and y and evaluate spline at those points
        spline_eval.append(spline_2d.eval2d(x[point], y[point]))

    # accurate values for comparison
    accurate = f(x, y)

    fig = plt.figure()
    plt.plot(x, accurate, 'b', label='Real')
    plt.plot(x, np.squeeze(spline_eval), 'r-.', label='Estimate from interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
def main():
    interpolate()

if __name__ == "__main__":
    main()