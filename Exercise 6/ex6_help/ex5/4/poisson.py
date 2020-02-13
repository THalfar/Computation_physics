from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

""" FYS-4096 Computational Physics """
""" Exercise 5, problem 3 """
""" Roosa Hytönen 255163 """


def jacobi(phi, h, rho, N):
    """ Function that implements the Jacobi update scheme as defined in lecture material
    """
    phi2 = phi.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            phi2[i, j] = 0.25*(phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] + h**2/8.8541878128e-12*rho[i, j])
    return phi2


def gauss_seidel(phi, h, rho, N):
    """ Function that implements the Gauss-Seidel update scheme as defined in lecture material
    """
    phi2 = phi.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            phi2[i, j] = 0.25*(phi[i+1, j] + phi2[i-1, j] + phi[i, j+1] + phi2[i, j-1] + h**2/8.8541878128e-12*rho[i, j])
    return phi2


def SOR(phi, h, rho, N, omega=1.8):
    """ Function that implements the simultaneous over relaxation relaxation scheme (SOR) as defined in lecture material
    """
    phi2 = phi.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            phi2[i, j] = (1-omega) * phi2[i, j] + (omega/4)*(phi[i+1, j] + phi2[i-1, j] + phi[i, j+1] + phi2[i, j-1] +
                                                             h**2/8.8541878128e-12*rho[i, j])
    return phi2


def main():
    """ Solving the Poisson equation in 2-dimensions using the Jacobi, Gauss-Seidel and SOR update schemes
    """

    """ Setting up phi matrices for all methods, using initial value of 1 in each case
    """
    N = 40
    phi_1 = np.zeros((N, N), dtype=float)
    phi_2 = np.zeros((N, N), dtype=float)
    phi_3 = np.zeros((N, N), dtype=float)
    phi_1[:, 0] = 1
    phi_2[:, 0] = 1
    phi_3[:, 0] = 1

    """ rho = 0, Laplace equation"""
    rho = np.zeros((N, N), dtype=float)
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    h = x[1]-x[0]

    reps = 50
    for i in range(reps):
        """ New values
        """
        phi_new1 = jacobi(phi_1, h, rho, N)
        phi_new2 = gauss_seidel(phi_2, h, rho, N)
        phi_new3 = SOR(phi_3, h, rho, N)
        """ Updating value to phi matrix
        """
        phi_1 = phi_new1.copy()
        phi_2 = phi_new2.copy()
        phi_3 = phi_new3.copy()

    """ Comparing maximum absolute differences of the three methods
    """
    print('Maximum difference of Jacobi and Gauss-Seidel method:', np.amax(abs(phi_1-phi_2)))
    print('Maximum difference of Jacobi and SOR method:', np.amax(abs(phi_1-phi_3)))
    """ Forming 3D plots for all cases
    """
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, phi_1, cmap=cm.cool)
    plt.title('Jacobi update scheme')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('$\Phi(x,y)$')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, phi_2, cmap=cm.summer)
    plt.title('Gauss-Seidel update scheme')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('$\Phi(x,y)$')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, phi_3, cmap=cm.magma)
    plt.title('SOR update scheme')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('$\Phi(x,y)$')

    plt.show()


if __name__ == "__main__":
    main()

