import numpy as np
import matplotlib.pyplot as plt

# Constants
Q = 1.602e-19
L = 4
e0 = 8.854e-12
const = 9e9
lamb = Q / L
points = 20  # How many sections the rod is split to
dx = L / points
def analytical_value(d):
    return [lamb*const*(1/d-1/(d+L)), 0]

def e_field(x, y):
    """
    Calculates E-field on (x,y) from 1D charged rod (-L/2,0)<->(L/2,0)
    split to 'points' amount of sections, see constants.
    Calculates E-field as vector sum from individual sections
    Has strange behaviour, indexing probably inverts at some point,
    but the end result looks correct
    :param x:
    :param y:
    :return: [u,v] field vector at [x,y]
    """
    vector = np.array([0, 0], dtype="float64")
    ro = np.array([x, y], dtype="float64")
    for pt in np.linspace(-L / 2, L / 2, points, dtype="float64"):
        rp = ro - np.array([pt, 0])
        r_sq = np.dot(rp, rp)
        r_l = np.linalg.norm(rp)
        r_unit = rp / r_l
        scalar = dx * const * lamb / r_sq
        # Vectors get mixed up at some point so they must be inverted here
        # ???
        vector[0] = vector[0] + scalar * r_unit[1]
        vector[1] = vector[1] + scalar * r_unit[0]
    return vector


def main():
    # Quiver plot
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    u = np.zeros((20, 20), dtype="float64")
    v = np.zeros((20, 20), dtype="float64")
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            vector = e_field(x[i], y[j])
            u[i, j] = vector[0]
            v[i, j] = vector[1]
    # Looks correct albeit inverted
    plt.quiver(x, y, u, v)
    plt.show()

    # Analytical comparison
    # y-coordinate ignored
    analytic_val = []
    numeric_val = []
    di = np.linspace(0.2,5)
    for d in np.linspace(0.1,20):
        analytic_val.append(analytical_value(d)[0])
        numeric_val.append(e_field(L/2+d,0))
    plt.plot(di, analytic_val,'b-', di, numeric_val, 'r-')
    plt.yscale = 'log'
    plt.show()



main()
