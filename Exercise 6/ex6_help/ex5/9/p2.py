import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def a(x0, t, E, B):
    """
    Acceleration of charged particle
    :param x0: State vector
    :param t: Time
    :param E: Electric field
    :param B: Magnetic field
    :return:
    """
    r0 = x0[0:3]
    v0 = x0[3:]
    xx = 0.0 * x0
    xx[0:3] = v0
    xx[3:] = E + np.cross(v0, B)
    return xx

def main():

    v0 = [0.1, 0.1, 0.1]
    r0 = [0.0, 0.0, 0.0]
    E = [0.05, 0, 0]
    B = np.transpose([0.0, 4.0, 0])

    t = np.linspace(0, 5, 100)
    x_coords = np.zeros_like(t)
    y_coords = np.zeros_like(t)
    z_coords = np.zeros_like(t)
    x0 = np.concatenate((r0, v0))  # State vector
    xx = odeint(a, x0, [0, 5], args=(E, B))

    print("Coordinates at t = 5:", xx[1][0:3], ", Velocity:", xx[1][3:])
    i=0
    for i in range(len(t)):
        xx = odeint(a, x0, [0, i], args=(E, B))
        x_coords[i] = xx[1][0]
        y_coords[i] = xx[1][1]
        z_coords[i] = xx[1][2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_coords, y_coords, z_coords)
    plt.show()

main()
