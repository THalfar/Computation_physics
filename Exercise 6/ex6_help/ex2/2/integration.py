from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import num_calculus as nc

""" FYS-4096 Computational Physics: Exercise 2 """
""" Author: Santeri Saariokari """

def integrate_a():
    f1 = lambda r: r**2 * np.exp(-2*r)
    f2 = lambda x: np.sin(x) / x
    f3 = lambda x: np.exp(np.sin(x**3))

    int1 = []
    int2 = []
    int3 = []
    int1_error = []
    int2_error = []
    int3_error = []

    grid_sizes = np.linspace(3, 50, 50, dtype=int)
    for points in grid_sizes:

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad
        # scipy.integrate.quad | Other Parameters :
        # limit: float or int, optional
        # An upper bound on the number of subintervals used in the adaptive algorithm.

        int1.append(integrate.quad(f1, 0, np.inf, limit=points)[0])


        # function 2 is not defined at 0, so we choose a "small value"
        # for the lower bound
        x2 = np.linspace(1e-10, 1, points)
        int2.append(integrate.simps(f2(x2), x2))

        x3 = np.linspace(0, 5, points)
        int3.append(integrate.simps(f3(x3), x3))


    plt.subplot(311).plot(grid_sizes, int1)
    plt.ylim([0,1])
    plt.subplot(312).plot(grid_sizes, int2)
    plt.subplot(313).plot(grid_sizes, int3)
    plt.tight_layout()

    print("a)")
    print("Numerical result: %.5f, Analytical result: %.5f" % (int1[-1], 0.25))
    print("Numerical result: %.5f, Analytical result: %.5f" % (int2[-1], 0.946083))
    print("Numerical result: %.5f, Analytical result: %.5f" % (int3[-1], 6.647272))
    plt.show()


def integrate_b():
    f = lambda x, y: x * np.exp(-np.sqrt(x**2 + y**2))
    grid_sizes = np.linspace(10, 200, 100, dtype=int)
    integrals = []
    for points in grid_sizes:
        x_values = np.linspace(0, 2, points)
        y_values = np.linspace(-2, 2, points)
        dx = x_values[1] - x_values[0]
        I = 0
        for x in x_values:
            values_at_fixed_x = f(x, y_values)
            I += integrate.simps(values_at_fixed_x, y_values) * dx
        integrals.append(I)

    fig = plt.figure()
    plt.plot(grid_sizes, integrals)
    plt.xlabel("points")
    plt.ylabel("Integral")

    print("b)")
    print("Numerical result: %.5f, Analytical result: %.5f" % (integrals[-1], 1.57347))
    plt.show()

def integrate_c():
    r_A = (-0.7, 0, 0)
    r_B = (0.7, 0, 0)
    norm = lambda x: np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    euclid = lambda x, y: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)
    psi = lambda r: np.exp(-norm(r)) / np.sqrt(np.pi)
    integrand = lambda R: abs(psi(np.subtract(R, r_A)))**2 / euclid(R, r_B)
    analytic_result = lambda r_A, r_B: (1-(1 + euclid(r_A, r_B)) * np.exp(-2 * euclid(r_A, r_B))) / euclid(r_A, r_B)

    grid_point_amounts = np.linspace(2, 70, 10, dtype=int)
    integrals = []
    for points in grid_point_amounts:
        grid_points = np.linspace(-7, 7, points)
        dx = dy = dz = grid_points[1] - grid_points[0]
        I = 0
        for x in grid_points:
            for y in grid_points:
                for z in grid_points:
                    r_i = (x, y, z)
                    I += integrand(r_i) * dx * dy * dz
        integrals.append(I)
    fig = plt.figure()
    plt.plot(grid_point_amounts, integrals)
    plt.xlabel("points")
    plt.ylabel("Integral")

    print("c)")
    print("Numerical result: %.5f, Analytical result: %.5f" % (integrals[-1], analytic_result(r_A, r_B)))
    plt.show()


def main():
    integrate_a()
    integrate_b()
    integrate_c()

if __name__=="__main__":
    main()
