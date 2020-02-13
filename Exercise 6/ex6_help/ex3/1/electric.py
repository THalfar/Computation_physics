import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

eps_0 = 8.8541878128e-12

def E_field(L, Q, r, points):
    """
    Calculates the electric field of a rod with charge Q and length L
    at point r, approximating the rod with "points" amount of point charges
    """
    field = np.array([0., 0.])
    dx = L / (points - 1)
    for dr in np.linspace(-L/2, L/2, points):
        r_0 = np.array([dr + dx/2, 0]) # center of point charge -> add dx/2
        r_tot = r - r_0
        field += dE(Q, L, r_tot) / points
    return field

def E_xy(L, Q, points):
    """
    Calculates and draws the electric field of a rod with charge Q and length L
    approximating the rod with "points" amount of point charges
    """
    grid_size = 20
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X,Y = np.meshgrid(x, y)

    E_x = np.zeros_like(X)
    E_y = np.zeros_like(Y)
    for j in range(20):
        for i in range(20):
            r = np.array((x[j], y[i]))
            field = E_field(L, Q, r, points)
            E_x[i, j] = field[0]
            E_y[i, j] = field[1]
    fig = plt.figure()
    plt.quiver(X, Y, E_x, E_y)
    plt.plot([-L/2, L/2], [0,0]) # rod
    plt.title('Electric field of uniformly charged rod')
    plt.show()

def E_test():
    """
    test function for calculating the electric field at single point
    """
    d=0.1
    L = 1
    Q = 1
    r = (L/2+d, 0)
    correct = E_analytic(L, Q, d)
    estimate = E_field(L, Q, r, 1000)
    print("Correct:  %.3f\nEstimate: %.3f\nRelative error: %.5f"
            % (correct, estimate[0], abs(estimate[0] - correct ) / correct))

def plot_convergence(grid_sizes, integrals):
    """
    Draws "integrals" as function of "grid_sizes"
    """
    fig = plt.figure()
    plt.plot(grid_sizes, integrals)
    plt.xlabel("Points")
    plt.ylabel("Integral")
    plt.show()

def dE(Q, L, r):
    """
    Electric charge of rod at point r
    """
    # nominator has r^3 because of the normalization of r to r_unit
    return 1/(4*np.pi*eps_0) * Q * r / (L * np.linalg.norm(r)**3)

def E_analytic(Q, L, d):
    """
    Electric charge of rod at distance d, measured from the end of rod
    """
    return 1/(4*np.pi*eps_0) * (1/d-1/(d+L)) * Q / L

def main():
    E_test()
    E_xy(1,1,100)

if __name__ == "__main__":
    main()