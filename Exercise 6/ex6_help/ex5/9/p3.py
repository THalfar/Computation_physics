import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def jacobi():
    """
    Solves Poisson equation with Jacobi method
    :return:
    """
    tolerance = 1.0e-6
    hx = hy = 25  # Grid spacing
    x = np.linspace(0, 1, hx)
    y = np.linspace(0, 1, hy)
    xx, yy = np.meshgrid(x, y)
    T = np.zeros_like(xx)  # set up matrix for calculating T values

    # Boundary values
    T[-1, :] = 0  # Bottom temperature
    T[0, :] = 0  # Top temperature
    T[:, 0] = 1  # Left temperature
    T[:, -1] = 0  # Right temperature

    # Plot of initial conditions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, T, rstride=1, cstride=1)
    plt.show()

    iterations = 0
    avg_diff = 1
    # Jacobi method
    while avg_diff > tolerance:
        avg_old = np.average(T)
        for i in range(1, hx - 1):  # 0, hy - 1
            for j in range(1, hy - 1):
                T[i, j] = (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]) / 4
        avg_diff = abs(avg_old - np.average(T))  # difference of averages to figure out when to stop iterating
        iterations += 1
    print("Jacobi iterations", iterations)
    # Plot of Jacobi method solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, T, rstride=1, cstride=1)
    plt.show()


def gauss_seidel():
    """
    Solves Poisson equation with Gauss-Seidel
    :return:
    """
    xmax = 1
    ymax = 1
    tolerance = 1.0e-6

    hx = hy = 25
    x = np.linspace(0, xmax, hx)
    y = np.linspace(0, ymax, hy)
    xx, yy = np.meshgrid(x, y)
    T = np.zeros_like(xx)

    # Boundary values
    T[-1, :] = 0  # Bottom temperature
    T[0, :] = 0  # Top temperature
    T[:, 0] = 1  # Left temperature
    T[:, -1] = 0  # Right temperature

    iterations = 0
    avg_diff = 1
    # Gauss-Seidel method
    while avg_diff > tolerance:
        avg_old = np.average(T)
        for i in range(1, hx - 1):
            for j in range(1, hy - 1):
                # For G-S method we also need values from the next T, so it's calculated twice
                Tnext = T
                Tnext[i, j] = (T[i + 1][j] + Tnext[i - 1][j] + T[i][j + 1] + Tnext[i][j - 1]) / 4
                T[i, j] = (T[i + 1][j] + Tnext[i - 1][j] + T[i][j + 1] + Tnext[i][j - 1]) / 4
        avg_diff = abs(avg_old - np.average(T))
        iterations += 1
    print("G-S iterations:", iterations)
    # Gauss-Seidel plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, T, rstride=1, cstride=1)
    plt.show()


def sor():
    """
    Solves Poisson equation with SOR method
    :return:
    """
    xmax = 1
    ymax = 1
    tolerance = 1.0e-6
    omega = 1.8

    hx = hy = 25
    x = np.linspace(0, xmax, hx)
    y = np.linspace(0, ymax, hy)
    xx, yy = np.meshgrid(x, y)
    T = np.zeros_like(xx)

    # Boundary values
    T[-1, :] = 0  # Bottom temperature
    T[0, :] = 0  # Top temperature
    T[:, 0] = 1  # Left temperature
    T[:, -1] = 0  # Right temperature

    iterations = 0
    avg_diff = 1
    # SOR method
    # Similar to G-S method, only with additional omega factor
    while avg_diff > tolerance:
        avg_old = np.average(T)
        for i in range(1, hx - 1):  # 0, hy - 1
            for j in range(1, hy - 1):
                Tnext = T
                Tnext[i, j] = (1 - omega) * T[i, j] + (
                            T[i + 1][j] + Tnext[i - 1][j] + T[i][j + 1] + Tnext[i][j - 1]) * omega / 4
                T[i, j] = (1 - omega) * T[i, j] + (
                            T[i + 1][j] + Tnext[i - 1][j] + T[i][j + 1] + Tnext[i][j - 1]) * omega / 4
        avg_diff = abs(avg_old - np.average(T))
        iterations += 1
    print("SOR iterations:", iterations)

    # Plot figure of SOR method solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, T, rstride=1, cstride=1)
    plt.show()


def main():
    jacobi()
    gauss_seidel()
    sor()


main()
