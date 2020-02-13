"""
Calculating the electric field caused by a charged 1D rod.
"""
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

def electric_field(x, y, z, points = 100, L=1, Q=0.01):

    # Define the physical constants
    charge_density = Q/L
    epsilon = 8.85418e-12

    # Function to integrate
    def E(l, x, y, z):
        r_squared = (x-l)**2 + y**2 + z**2
        direction = 1 / np.sqrt(r_squared) * np.array([x-l, y, z])
        return 1/(4*np.pi*epsilon)*charge_density/r_squared*direction

    l = np.linspace(-L/2, L/2, points)
    # This is required, because otherwise the integration will try to
    # use l as an array in E, while it should be a float.
    field = np.zeros((len(l),3))
    for i in range(points):
        field[i] = (E(l[i], x, y, z))

    # Integrate the fields caused by all the points in the rod.
    int_x = simps(field[:, 0], l)
    int_y = simps(field[:, 1], l)
    int_z = simps(field[:, 2], l)

    return [int_x, int_y, int_z]

def main():


    # Test the fuction against an analytical solution.
    L = 1
    Q = 0.01
    charge_density = Q / L
    epsilon = 8.85418e-12
    d = 10

    analytical_answer = charge_density / (4*np.pi*epsilon) * (1/d - 1/(d + L))
    numerical_answer = electric_field(L/2 + d, 0, 0)[0]

    print("analytical answer:", analytical_answer)
    print("numerical answer:", numerical_answer)


    # Plot the field.

    # Turns out to be very hard when electric_field() can only take floats for the coordinates...



main()