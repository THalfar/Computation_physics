"""
This file contains the solution to problems 1 and 2 in exercise set 3 in 
Computational Physics course.

author: Juha Teuho
"""

from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
from linear_interp import linear_interp

def fun(x,y):
    """Function needed for problems 1 and 2
    
    Args:
        x (numpy array): x variable
        y (numpy array): y variable
    
    Returns:
        function handle: The function in terms of x and y
    """
    return (x+y)*np.exp(-np.sqrt(x**2+y**2))

def evaluate2DIntegral(fun, xMin, xMax, yMin, yMax, xGridPoints, yGridPoints):
    """This function evaluates the 2-dimensional integral using Simpson's method
    from scipy.integrate-library.
    
    Args:
        fun (function handle): integrand
        xMin (float): lower limit for first integration variable
        xMax (float): upper limit for first integration variable
        yMin (float): lower limit for second integration variable
        yMax (float): upper limit for second integration variable
        xGridPoints (int): number of grid points in direction of first integration variable
        yGridPoints (int): number of grid points in direction of second integration variable
    
    Returns:
        float: value of the integral
    """

    # Create evenly spaced grid.
    x = np.linspace(xMin,xMax,xGridPoints)
    y = np.linspace(xMin,xMax,yGridPoints)
    [X,Y] = np.meshgrid(x,y)
    F = fun(X,Y)
    # Perform integration in x first.
    Idx = simps(F,x)
    I = simps(Idx,y)
    return I

def test_evaluate2DIntegral():
    """
    Test 2D integral function for a certain function.
    Study the convergence of the integral with respect to number of grid points.
    """

    # Variable limits in problem 1
    xmin = 0
    xmax = 1
    ymin = -2
    ymax = 2

    smallestGrid = 10 # 10 x 10 grid
    largestGrid = 50 # 50 x 50 grid

    # Loop through all grids in steps of 1.
    integrals = np.empty(largestGrid-smallestGrid+1)
    for xPoints in range(smallestGrid,largestGrid+1):
        # symmetrical grid
        yPoints = xPoints
        integral = evaluate2DIntegral(fun,xmin,xmax,ymin,ymax,xPoints,yPoints)
        integrals[xPoints-10] = integral

    # Plot and save the convergence of the integral when grid size grows.
    plt.style.use('seaborn-bright')
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    ax.plot(np.linspace(smallestGrid,largestGrid,largestGrid-smallestGrid+1),integrals)
    ax.set_xlabel('n in (n x n) grid')
    ax.set_ylabel('Integral value')
    ax.set_title('Convergence of integral in problem 1 \n with respect to grid points')
    fig.tight_layout(pad=1)
    fig.savefig('problem1.pdf', dpi=200)
    plt.show()

def Line(x, k):
    """Define line for certain points when slope is known
    
    Args:
        x (numpy array): x-values for the line
        k (TYPE): Description
    
    Returns:
        numpy array: Line values at x.
    """
    return k*x


def interpolate():
    """ Solution to problem 2. Interpolate the "experimental data" to get the
    approximation of the value on a line.
    """

    # Define the boundaries and grid
    xMin = -2
    xMax = 2
    yMin = -2
    yMax = 2
    xGridPoints = 30
    yGridPoints = 30

    # Make the grid
    x = np.linspace(xMin,xMax,xGridPoints)
    y = np.linspace(xMin,xMax,yGridPoints)
    [X,Y] = np.meshgrid(x,y)

    # Calculate "experimental data"
    experimentalData = fun(X,Y)

    # Do the interpolation. The interpolation must be done in a for-loop one 
    # point by one, since the interpolation function would use grid if the input
    # parameters were arrays.
    lin2d=linear_interp(x=x,y=y,f=experimentalData,dims=2)
    xs = np.linspace(0,2,num=100)

    # Loop over each point on line.
    slope = np.sqrt(1.75)
    interpolation = []
    for x in xs:
        y = Line(x, slope)
        Z = lin2d.eval2d(x,y)
        interpolation.append(Z[0][0])

    # analytical solution
    analytical = fun(xs,Line(xs,slope))

    # Plot numerical estimations and analytical solution
    plt.style.use('seaborn-bright')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs,analytical, label='Analytical')   
    ax.plot(xs,interpolation,'r--', label='Interpolation')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x,\sqrt{1.75}x)$')
    ax.set_title('Interpolation of $f(x,y)$ on line $y=\sqrt{1.75}x$')
    ax.legend()
    fig.tight_layout(pad=1)
    fig.savefig('problem2.pdf', dpi=200)
    plt.show()

def main():
    # perform the tests
    test_evaluate2DIntegral()
    interpolate()

if __name__=="__main__":main()