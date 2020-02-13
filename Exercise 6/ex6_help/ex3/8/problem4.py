import numpy as np
import matplotlib.pyplot as plt

# vacuum permittivity
epsilon = 8.8541878128*10e12

def funE(r,l, dx):
    """
    Function for calculating electric field at point
    :param r: distance from point
    :param l: charge density in 1d
    :param dx: piece of x coordinate
    :return: fun value
    """
    return 1/(4*np.pi*epsilon)*l*dx/r**2

def calculate_distance(x1,y1,x2,y2):
    """
    calculetes distance in 2D between two points
    :param x1: x coordinate of first point
    :param y1: y coordinate of first point
    :param x2: x coordinate of second point
    :param y2: y coordinate of second point
    :return: distance between points
    """
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def calculate_electric_field(L,Q,gaps,x,y):
    """
    Calculate electric field at given point from charged rod. Returns electric field for x and y direction
    separately.
    :param L: length of the rod
    :param Q: charge
    :param gaps: number of gaps to calculate
    :param x: x coordinate of point
    :param y: y coordinate of point
    :return: electric field in x and y direction
    """
    # calculating parameters needed
    # charge density in 1d
    l=Q/L
    xx = np.linspace(-2/L, 2/L, gaps)
    yy = np.zeros(len(xx))
    dx = xx[1]-xx[0]
    Ey = 0;
    Ex = 0
    # looping through L with dx
    for i in range(len(xx)-1):
        r = calculate_distance(x,y,xx[i]+dx/2,yy[i]+dx/2)
        # formulas for x and y coordinates are simply cos(theta) and sin(theta) for the point compared to dx
        Ey = Ey + funE(r, l, dx)*y/r
        Ex = Ex + funE(r, l, dx)*(x-xx[i]-dx/2)/r
    return Ex, Ey

def from_x_y_to_E(Ex, Ey):
    """
    calculate magnitude of electric field from x and y components
    :param Ex: Electric field in x direction
    :param Ey: Electric field in y direction
    :return: Electric field
    """
    return np.sqrt(Ex**2+Ey**2)

def analytical_fun(L,Q,d):
    """
    analytically calculates electric field when y=0
    :param L: length of the rod
    :param Q: electric charge
    :param d: distance from end of the rod
    :return: Electic field
    """
    l=Q/L
    return l/(4*np.pi*epsilon)*(1/d-1/(d+L))

def vector_map_electric_field():
    """
    plot vectors of electric field at given grid points
    """
    x = np.linspace(-2,2,20)
    y = np.linspace(-2,2,20)
    X, Y = np.meshgrid(x,y)
    u, v = calculate_electric_field(2, 1.6 * 10e-19, 1000, X, Y)

    # creating new figure
    fig1d = plt.quiver(X, Y, u, v)
    plt.show()

def main():
    Ex, Ey = calculate_electric_field(2, 1.6 * 10e-19, 1000, 5, 0)
    print("numerical at (5,0):  ", from_x_y_to_E(Ex, Ey))
    print("analytical at (5,0): ", analytical_fun(2,1.6*10e-19,4))
    vector_map_electric_field()

if __name__ == "__main__":
    main()