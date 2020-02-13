"""
FYS-4096 Computational Physics
Exercise 3
Problem 4

Charged rod's electric field problem
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def electric_field(x, y, Q=1.602e-19, L=1):

    # Integration "grid" in 1D for integration over the rod from -L/2 to L/2
    integral_points = np.linspace(-L/2, L/2, 1000)

    r = np.sqrt((x-integral_points)**2 + (y)**2)
    r_vec = [(x-integral_points)/r, y/r]

    const = 1/(4*np.pi*8.854e-12)*Q/L/r**2

    dex = np.array(const * r_vec[0])
    ex = simps(dex, integral_points)

    dey = np.array(const * r_vec[1])
    ey = simps(dey, integral_points)

    return ex, ey


def analytical_y_axis_solution(x, Q, L):
    lam = Q/L
    d = x-L/2
    ex = lam/(4*np.pi*8.854e-12) * (1/d-1/(d+L))
    ey = 0

    return ex, ey

def test_vs_analytical(d, L, Q):
    """
    Test case against known analytical solution / answer
    :param d: distance from the L/2 end of the rod
    :param L: length of the rod
    :param Q: charge of the rod
    :return: Prints whether the test passed, no return value
    """
    try:
        print(bcolors.OKBLUE+"Testing for r=(L/2+d, 0):")


        print("Test parameters: d = {:.2f}, L = {:.2f}, Q = {:.3e}".format(d, L, Q))

        numerical = electric_field(L / 2 + d, 0, Q=Q, L=L)
        analytical = analytical_y_axis_solution(L / 2 + d, Q=Q, L=L)

        print(bcolors.WARNING+"Analytical: {}, Numerical: {}".format(analytical, numerical))

        err = np.abs(numerical[0]-analytical[0])/analytical[0]
        if(err <= 1e-8):
            print(bcolors.OKGREEN+"Error compared to analytical answer: {:.8f} %".format(err * 100))
        else:
            print(bcolors.FAIL+"Error compared to analytical answer: {:.8f} %".format(err * 100))

        assert err <= 1e-8
        print(bcolors.OKGREEN+"Test passed.")
    except AssertionError as e:
        print(bcolors.FAIL+"The test failed.")

    print(bcolors.ENDC)


def main():
    d = 0.5
    L = 1
    Q = 1.602e-19

    # Test case: r = (L/2+d, 0), e.g. on x-axis
    test_vs_analytical(d, L, Q)

    # Calculating the field on 50x50 grid, -L to L on both axes
    xs = np.linspace(-L, L, 50)
    ys = np.linspace(-L, L, 50)
    x, y = np.meshgrid(xs, ys)

    ex = np.zeros_like(x)
    ey = np.zeros_like(y)

    for i in range(ex.shape[0]):
        for j in range(ex.shape[1]):
            ex[i, j], ey[i, j] = electric_field(x[i,  j], y[i, j])

    fig = plt.figure()
    # speed = np.sqrt(ex**2 + ey**2)
    # lw = 5 * speed / speed.max()
    # plt.streamplot(x, y, ex, ey, linewidth=lw)
    plt.quiver(x, y, ex, ey)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__=="__main__":
    main()
