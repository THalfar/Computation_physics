"""
Calculating and plotting the electron density along two lines. 
Not working correctly.
"""

from read_xsf_example import read_example_xsf_density
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt

rho, lattice, grid, shift = read_example_xsf_density("dft_chargedensity2.xsf")

# define the points on the first line.
x = np.linspace(-1.4466, 1.1461, 500)
# fix x so that it does not go uotside the unit cell using periodicity.
for i in range(len(x)):
    if x[i] < 0:
        x[i] = x[i] + 5.75200
y = np.linspace(1.3073, 3.1883, 500)
z = np.linspace(3.2115, 1.3542, 500)

# Points for interpolation.
xi = np.linspace(0, lattice[0, 0], grid[0])
yi = np.linspace(0, lattice[1, 1], grid[1])
zi = np.linspace(0, lattice[2, 2], grid[2])


# Interpolate the density.
interp = rgi((xi, yi, zi), rho)

# Get the interpolated density in the points on the line.
points_density = np.zeros(500)
for i in range(500):
    points_density[i] = interp((x[i], y[i], z[i]))

# Plot the density.
plt.plot(y, points_density)
plt.xlabel("Distance in the direction of the second lattice vector (Å)")
plt.ylabel("Electron density")
plt.show()