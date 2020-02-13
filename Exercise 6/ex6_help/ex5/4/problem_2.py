import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.integrate import odeint

""" FYS-4096 Computational physics """
""" Exercise 5 Problem 2 """
""" Roosa Hytönen 255163 """


def f(vec, t, a, b):
    """ Equation of motion for the charged particle under the influence of the Lorentz force. Input vector vec composed
        of x, y, z coordinates (r0 vector) and vx, vy, vz (v0 vector) velocity vectors. Returns an array containing time
        derivatives of position and velocity.
    """
    dydt = np.zeros(6)
    """ Setting velocity components as time derivatives of position
    """
    dydt[0:3] = vec[3:]
    """ Setting time derivatives of velocity
    """
    vr = (a + np.cross(vec[3:], b))
    dydt[3:] = vr
    return np.array(dydt)


def main():
    t = np.linspace(0, 5, 101)
    """ Initial values used to solve the EOM of a charged particle under the influence of the Lorentz force with the 
        odeint scipy solver for ordinary differential equations.
    """
    vec0 = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    a = np.array([0.05, 0, 0])  # qE/m
    b = np.array([0, 4.0, 0])  # qB/m
    """ Solving the EOM
    """
    sol = odeint(f, vec0, t, args=(a, b))
    """ Plotting the trajectory of the charged particle in 3D
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'pink', linewidth=2)
    ax.set_title(r'Trajectory of the particle for $t\in [0,5]$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    """ Calculating the velocity vector at t=5, that is, simply extracting the last values obtained for vx, vy, vz from
        the solution matrix
    """
    sol_t5 = sol[-1]
    v_vec = sol_t5[3:]
    print('Velocity vector at t = 5:', v_vec)


if __name__ == "__main__":
    main()
