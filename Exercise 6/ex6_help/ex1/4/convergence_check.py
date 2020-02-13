"""
This file contains the code for evaluating and plotting the convergences
for absolute errors of first_derivative, second_derivative and riemann_sum
functions in file num_calculus.py

This code is part of the course "Computational Physics"

Juha Teuho
"""

from num_calculus import first_derivative, second_derivative, riemann_sum

import numpy as np
import matplotlib.pyplot as plt

# Let's use a cool style sheet for plotting.
plt.style.use('fivethirtyeight')

# We will evaluate the errors using sin(x) as the test function. We also need
# to know the first and second derivatives and the integral of this function.
def test_function(x):
    return np.sin(x)

def der_test_function(x):
    return np.cos(x)

def sec_der_test_function(x):
    return -np.sin(x)

def int_test_function(xmin, xmax):
    return -np.cos(xmax) + np.cos(xmin)

"""
Let's first calculate the errors for all of the three cases
"""

# For the derivatives, we need to vary the grid spacing.
# We will also use a single value for variable x.
dxs = np.linspace(0.001,1,num=1001)
x = 2.0

# For the integral, we need to define the integration limits.
# The range of iterations is also needed for the x-axis of the plot.
xmin = 0
xmax = np.pi/2
rs_x = np.linspace(2,50,num=48)

# Analytical solutions for each function
analytical_fd = der_test_function(x)
analytical_sd = sec_der_test_function(x)
analytical_rs = int_test_function(xmin,xmax)

errors_fd = [] # errors for first derivative
errors_sd = [] # errors for second derivative
errors_rs = [] # errors for riemann_sum integration

# Calculate errors for derivatives
for dx in dxs:
    numerical_fd = first_derivative(test_function, x, dx)
    error_fd = np.abs(numerical_fd-analytical_fd)

    numerical_sd = second_derivative(test_function, x, dx)
    error_sd = np.abs(numerical_sd-analytical_sd)

    errors_fd.append(error_fd)
    errors_sd.append(error_sd)

# Calculate errors for integration
for intervals in range(2,50):
	x_range = np.linspace(xmin,xmax,num=intervals)
	numerical_rs = riemann_sum( x_range, test_function )
	error_rs = np.abs(numerical_rs-analytical_rs)
	errors_rs.append(error_rs)	


# Plot first derivative
fig_fd = plt.figure(1)
ax = fig_fd.add_subplot(111)
ax.plot(dxs,errors_fd,label=r'$f(x)=\sin(x)$')
ax.legend(loc=0)
ax.set_title('Error convergence of\nfirst derivative function')
ax.set_xlabel(r'grid spacing')
ax.set_ylabel(r'absolute error')
fig_fd.tight_layout(pad=1)
fig_fd.savefig('fd_convergence.pdf', dpi=200)
plt.show()

# Plot second derivative
fig_sd = plt.figure(2)
ax = fig_sd.add_subplot(111)
ax.plot(dxs,errors_sd,label=r'$f(x)=\sin(x)$')
ax.legend(loc=0)
ax.set_title('Error convergence of\nsecond derivative function')
ax.set_xlabel(r'grid spacing')
ax.set_ylabel(r'absolute error')
fig_sd.tight_layout(pad=1)
fig_sd.savefig('sd_convergence.pdf', dpi=200)
plt.show()

# Plot riemann sum
fig_fd = plt.figure(3)
ax = fig_fd.add_subplot(111)
ax.plot(rs_x,errors_rs,label=r'$f(x)=\sin(x)$')
ax.legend(loc=0)
ax.set_title('Error convergence of\nRiemann sum')
ax.set_xlabel(r'Number of intervals')
ax.set_ylabel(r'Absolute error')
fig_fd.tight_layout(pad=1)
fig_fd.savefig('rs_convergence.pdf', dpi=200)
plt.show()
