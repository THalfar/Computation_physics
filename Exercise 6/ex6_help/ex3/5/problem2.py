from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from exercise3.linear_interp import linear_interp

"""CompPhys exercise 3 problem 2"""

def linear_index(grid, x0):
    """ COPYPASTE FROM EXERCISE 2, turns out to be broken
    Assuming linear grid, distance between grid points is x[i+1]-x[i],
    thus arbitrary x0 is closest to the point x[j] ~ x0/(x[i+1]-x[i])
    if this isn't the closest point, fallback method is used
    :param grid: np.linspace() grid
    :param x0: float or int inside the bounds of the grid
    :return: int index where x[i] <= x0 < x[i+1]
    """
    l = np.abs(grid[2]-grid[1])
    index = int(x0//l)  # // floor division
    #if np.abs(x0-grid[index]) < l: # only works with monotonous grid TODO
    return index
    #else:
    #    print("Something happened, raising error")
    #    raise Exception("linear_index failed")

def make_exp_data():
    """
    Makes experimental data as per instructions,
    immutable as a precaution
    :return:
    """
    grid_pts = 30
    x0 = np.linspace(-2,2,grid_pts)
    y0 = np.linspace(-2,2,grid_pts)
    val = np.empty([grid_pts,grid_pts])
    for i in range(grid_pts):
        for j in range(grid_pts):
            val[i,j] = (x0[i]+y0[j])*np.exp(-1*(np.sqrt(x0[i]**2+y0[j]**2)))
    # Immutable
    val.flags.writeable = False
    return val, x0, y0

def exact_values(data, x, y):
    """
    Returns values on line y=x*sqrt(1.75) as [x, f(x,y)] pair
    Index finding is very broken and relies on magic numbers
    :param data: f(x,y) grid
    :param x:
    :param y:
    :return: [x, f(x,y)]
    """
    values = []
    xx = []
    x_0 = np.size(x)//2
    x_1 = np.size(x)-4 #magic number
    for xi in range(x_0, x_1):
        yi = x[xi]*np.sqrt(1.75)
        yi = linear_index(y, yi)+15 # index finding broken offset
        values.append(data[xi,yi])
        xx.append(x[xi])
    return [xx, values]

def linear_interpolation(data, x, y):
    """
    Interpolates data on y=x*sqrt(1.75) line using
    function from help files
    :param data:
    :param x:
    :param y:
    :return: interpolated [x, f(x,y)]
    """
    lin2d = linear_interp(x=x,y=y,f=data, dims=2)
    xi = np.linspace(0,1.5,100)
    yi = xi*np.sqrt(1.75)
    coords = zip(xi,yi)
    values = []
    for point in coords:
            values.append(float(lin2d.eval2d(*point)))
    return [xi, values]

def spline_interp(data, x, y):
    """
    Interpolates data on y=x*sqrt(1.75) line
    uses RectBivariateSpline function from scipy
    :param data:
    :param x:
    :param y:
    :return: interpolated [x, f(x,y)]
    """
    lin2d = RectBivariateSpline(x, y, data)
    xi = np.linspace(0, 1.5, 100)
    yi = xi * np.sqrt(1.75)
    coords = zip(xi, yi)
    values = []
    for point in coords:
        values.append(float(lin2d(*point)))
    return [xi, values]

def main():
    data, x, y = make_exp_data()
    exact_val = exact_values(data, x, y)
    lin_interp_val = linear_interpolation(data, x, y)
    spline_interp_val = spline_interp(data, x, y)
    # Original red points, linear blue line, spline green line
    plt.plot(exact_val[0],exact_val[1],'ro', lin_interp_val[0], lin_interp_val[1], 'b-', spline_interp_val[0], spline_interp_val[1], 'g-')
    plt.show()


main()