import numpy as np
from exercise2.linear_interp import *

def fun1(x,y):
    """
    function to "create" experimental data
    :param x: x coordinates
    :param y: y coordinates
    :return fun values
    """
    return (x+y)*np.exp(-np.sqrt(x**2+y**2))

def fun2(x):
    """
    function for required line
    :param x: x coordinates
    :return: fun values
    """
    return np.sqrt(1.75)*x

def data_points_for_line(fun):
    """
    calculates data points of given line
    :param fun: function of the line
    :return: allowed function values
    """
    # what x coordinate reaches max y coordinate on grid
    end = 2/np.sqrt(1.75)
    x = np.linspace(0,end,100)
    y = fun2(x)
    return x,y

def linear_interpolation():
    """

    :return:
    """
    # coordinates
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    # creates 30x30 grid for data points on given function
    X, Y = np.meshgrid(x, y)
    # interpolation
    Z = fun1(X,Y)
    lin2d = linear_interp(x=x, y=y, f=Z, dims=2)
    # points on live we are using and function values at these points
    x,y = data_points_for_line(fun2)
    z = fun1(x,y)
    # creating new figure
    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)
    # plotting, only taking diagonal elements form eval2d since it produces matrix of all
    # combinations of x and y.
    ax1d.plot(x, lin2d.eval2d(x,y).diagonal(), label= "interpolation")
    ax1d.plot(x, z, label = "real values")
    # plot legend
    ax1d.legend(loc=0)

    plt.show()

def main():
    linear_interpolation()

if __name__ == "__main__":
    main()