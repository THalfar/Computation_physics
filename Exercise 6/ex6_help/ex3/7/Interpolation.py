"""
Interpolating a function.
"""
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

# Define the function to interpolate
def f(x, y): return (x + y) * np.exp(-1*np.sqrt(x**2 + y**2))

# Grid axes used for the interpolation
x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)

# Interpolate points at y = sqrt(1.75)
points = np.zeros(len(x))
for i in range(len(x)):
    rep = splrep(y, f(x[i], y))
    points[i] = splev(np.sqrt(1.75), rep)

# Interpolate the new 1d array to get 100 points between 0 and 2
final_points_x = np.linspace(0, 2, 100)
rep = splrep(x, points)
final_points = splev(final_points_x, rep)

def main():

    # determine the real values from the function
    real_values = f(final_points_x, np.sqrt(1.75))

    # Plot the interpolated values and the real values.
    plt.plot( final_points_x, real_values, 'o-', final_points_x, final_points, 'o')
    plt.show()



main()