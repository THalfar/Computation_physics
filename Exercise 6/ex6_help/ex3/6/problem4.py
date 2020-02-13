"""
Solution to problem 4 of exercise set 3 in course "Computational physics"

author: Juha Teuho

Attributes:
    EPSILON0 (float): Permittivity of free space
"""

import numpy as np
import matplotlib.pyplot as plt

EPSILON0 = 1.0

def rod_Efield(x,y,z, rod, Q):
    """
    Calculates the electric field by the charged rod at any point in space.
    The rod is 1-dimensional along x-axis.
    
    Args:
        x (numpy.ndarray): grid x coordinates
        y (numpy.ndarray): grid y coordinates
        z (numpy.ndarray): grid z coordinates
        rod (numpy.linspace): rod grid
        Q (float): electric charge
    
    Returns:
        Ex (float): x-component of electric field at point r
        Ey (float): y-component of electric field at point r
        Ez (float): z-component of electric field at point r
    """

    # Calculate lenght of rod L, width of grid interval dx and charge density ld
    L = rod[-1]-rod[0]
    dx = rod[1]-rod[0]
    ld = Q/L

    # Loop through every interval and calculate the differential electric field
    Ex = .0
    Ey = .0
    Ez = .0
    for i in range(1,len(rod)):

        # Coordinate in grid for current interval block
        x0 = rod[0]+(i-1/2)*dx
        y0 = 0
        z0 = 0

        # Distance from point r to current interval block
        r_scal = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)

        # Unit vectors
        x_unit = (x-x0)/r_scal
        y_unit = (y-y0)/r_scal
        z_unit = (z-z0)/r_scal

        # Calculate electric field components
        dE = 1/(4*np.pi*EPSILON0)*(ld*dx)/(r_scal**2)
        Ex += dE*x_unit
        Ey += dE*y_unit
        Ez += dE*z_unit 

    return Ex, Ey, Ez

def analytical(ld,d,L):
    """
    Analytical solution for electric field vector by charged rod at point
    r = [L/2+d, 0, 0] where L is length of the rod and d is an arbitrary 
    distance
    
    Args:
        ld (float): charge density
        d (float): arbitrary distance
        L (float): length of the charged rod
    
    Returns:
        numpy.ndarray: electric field vector at point r
    """
    Ex = ld / (4*np.pi*EPSILON0) * (1/d - 1/(d+L))
    Ey = .0
    Ez = .0
    return Ex,Ey,Ez

def test_rod_Efield():
    """
    Test function for rod_Efield function
    """

    # Charged rod from x=-5 to x=5, 100 grid points
    rod = np.linspace(-5,5,100)
    L = (rod[-1]-rod[0]) # length of rod
    d = 5            # arbitrary distance for analytical calculation
    Q = 1            # electric charge
    ld = Q/L         # charge density
    # coorinates of the point
    x = L/2+d
    y = 0
    z = 0

    max_error = 0.0001 # error tolerance

    # Calculate numerial and analytical solutions and compare
    Ex_num, Ey_num, Ez_num = rod_Efield(x,y,z,rod,Q)
    Ex_ana, Ey_ana, Ez_ana = analytical(ld,d,L)
    num = [Ex_num, Ey_num, Ez_num]
    ana = [Ex_ana, Ey_ana, Ez_ana]
    print("Numerical: ", num)
    print("Analytical: ", ana)
    error = abs(np.linalg.norm(num)-np.linalg.norm(ana))

    if error < max_error:
        print("rodEfield test successfull")
    else:
        print("rodEfield test failed")


def Efield_2D(xGrid,yGrid,rodGrid,Q):
    """
    Calculate electric field on xy-plane and make quiver-plot from it.
    
    Args:
        xGrid (numpy.ndarray): x points for grid
        yGrid (numpy.ndarray): y points for grid
        rodGrid (numpy.ndarray): rod grid
        Q (float): electric charge
    
    Returns:
        Ex (numpy.ndarray): x-component of electric field on grid points
        Ey (numpy.ndarray): y-component of electric field on grid points
        Ez (numpy.ndarray): z-component of electric field on grid points
    """

    # Calculate electric field on area determined by grid
    [X,Y] = np.meshgrid(xGrid,yGrid)
    Z = np.zeros(X.shape)
    Ex, Ey, Ez = rod_Efield(X,Y,Z,rodGrid,Q) 
    return Ex, Ey, Ez

def test_Efield_2D():
    """
    Test function for Efield_2d
    """

    # Charged rod with length L and charge Q
    L = 2
    rod = np.linspace(-L/2,L/2,100)
    Q = 1

    # Make 20x20 grid from -2 to 2 in both x and y directions
    xgrid = np.linspace(-2,2,20)
    ygrid = np.linspace(-2,2,20)

    # Calculate electric field components
    Ex, Ey, Ez = Efield_2D(xgrid,ygrid,rod,Q)

    # Make quiver plot
    plt.style.use('seaborn-bright')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(xgrid,ygrid,Ex,Ey,width=0.007)
    ax.set_title('Electric field from charged rod (L = 2)')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    fig.tight_layout(pad=1)
    fig.savefig('quiverplot.pdf', dpi=200)
    plt.show()

def main():
        # perform the tests
        test_rod_Efield()
        test_Efield_2D()

if __name__=="__main__":main()
