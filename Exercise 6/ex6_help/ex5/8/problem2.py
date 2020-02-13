import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


def traj(y, t, qE_per_m_vec, qB_per_m_vec):
    """
        Function for diff eq. solver. Represents a charged particle in electric and magnetic field.
    :param y: y[0:2] = position components, y[3:5] = velocity components
    :param t: parameter for odeint solver
    :param qE_per_m_vec: vector describing the acceleration due to electric field
    :param qB_per_m_vec: vector describing the acceleration due to magnetic field
    :return: derivatives of each component in y
    """

    velx = y[3]
    vely = y[4]
    velz = y[5]

    dposx = velx
    dposy = vely
    dposz = velz

    dvel = qE_per_m_vec + np.cross([velx, vely, velz], qB_per_m_vec)  # lorentz force
    dvelx = dvel[0]
    dvely = dvel[1]
    dvelz = dvel[2]

    return [dposx, dposy, dposz, dvelx, dvely, dvelz]


def main():
    qE_per_m_vec = [0.05, 0., 0.]  # vector representing the acceleration due to electric field
    qB_per_m_vec = [0., 4.0, 0.] # vector representing the acceleration due to magnetic field

    # initial conditions, first three are x, y and z positions, last three are x, y and z velocities
    y0 = [0., 0., 0., 0.1, 0.1, 0.1]

    t = np.linspace(0, 5, 1000)  # time range to solve for
    sol = odeint(traj, y0, t, args=(qE_per_m_vec, qB_per_m_vec))

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:,0], sol[:,1], sol[:,2])  # plot x, y and z position
    title("Trajectory of a charged particle in electric and magnetic fields.")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # get final position and velocity
    vel_end = sol[-1, 3:6]
    pos_end = sol[-1, 0:3]
    print("\nPosition at t = " + str(t[-1]) + "s is " + str(pos_end))
    print("Velocity at t = " + str(t[-1]) + "s is " + str(vel_end))

    show()


if __name__ == "__main__":
    main()

