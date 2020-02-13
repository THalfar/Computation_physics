"""
Calculating and plotting the electron density along a line.
"""

from read_xsf_example import read_example_xsf_density
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt

rho, lattice, grid, shift = read_example_xsf_density("dft_chargedensity1.xsf")

# define the points on the line.
x = np.linspace(0.1, 4.45, 500)
y = np.linspace(0.1, 4.45, 500)
z = 2.8528

# Points for interpolation.
xi = np.linspace(0, lattice[0, 0], grid[0])
yi = np.linspace(0, lattice[1, 1], grid[1])
zi = np.linspace(0, lattice[2, 2], grid[2])

# Interpolate the density.
interp = rgi((xi, yi, zi), rho)

# Get the interpolated density in the points on the line.
points_density = np.zeros(500)
for i in range(500):
    points_density[i] = interp((x[i], y[i], z))

# Plot the density. The distance on the line is sqrt(x^2 + y^2) = sqrt(2*x^2)
plt.plot(np.sqrt(2*x**2), points_density)
plt.xlabel("Distance on the line (Å)")
plt.ylabel("Electron density")
plt.show()