import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D

def lorentz_force(v,t,a,b):
    """
    function for calculating acceleration from lorentz force
    :param v: velocity
    :param t: time
    :param a: qE/m
    :param b: qB/m
    :return: function values in array
    """
    # F=ma=q(E+v*B) => dv/dt=qE/m+v*qB/m=a+v*b
    dvdt = a + np.cross(v,b)
    return np.array(dvdt)

def position_from_speed(v,t,r0):
    """
    integrates position from array of speed and time
    :param v: velocity
    :param t: time
    :param r0: starting position
    :return: x,y,z values for position at given times
    """
    x = np.zeros(len(v[:,0]))
    y = np.zeros(len(v[:, 1]))
    z = np.zeros(len(v[:, 2]))
    x[0] = r0[0]
    y[0] = r0[1]
    z[0] = r0[2]
    # integrating from 0 to time for every step of time
    for i in range(1,len(v[:,0])):
        x[i] = simps(v[:i ,0], t[:i]) + r0[0]
        y[i] = simps(v[:i, 1], t[:i]) + r0[0]
        z[i] = simps(v[:i, 2], t[:i]) + r0[0]
    return x,y,z

def solve_diff_eq():
    """
    solves diff equation with numpys odeint
    :return: velocity and position
    """
    # starting values
    a = [0.05, 0, 0]
    b = [0, 4, 0]
    t = np.linspace(0,5,100)
    v0 = [0.1, 0.1, 0.1]
    r0 = [0.0, 0.0, 0.0]
    # velocity form odeint
    v = odeint(lorentz_force, v0, t, args=(a, b))
    # position from own function
    x,y,z = position_from_speed(v,t, r0)
    r = [x, y, z]
    return v, r

def main():
    # plot results
    v, r = solve_diff_eq()
    print("v at t=5: ", v[99])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[0], r[1], r[2])
    plt.show()

if __name__ == "__main__":
    main()