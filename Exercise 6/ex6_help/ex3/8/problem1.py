from scipy.integrate import simps
import numpy as np

def fun(x,y):
    """
    function for integration
    :param x: x coordinate
    :param y: y coordinate
    :return: function value at given coordinates
    """
    return (x+y)*np.exp(-np.sqrt(x**2+y**2))

def double_integral(fun,x,y):
    """
    calculates double integral over given coordinates
    :param fun: function for integration
    :param x: x coordinates
    :param y: y coordinates
    :return: integral value
    """
    first_integral = np.zeros(len(y))
    # calculating integrals in x direction for every y coordinate
    for i in range(len(y)):
        first_integral[i]=simps(fun(y[i],x),x)
    # integrals for y coordinates from already integrated x coordinates
    return simps(fun(y,first_integral),y)

def test_double_integral():
    """
    testing with different grids
    """
    x = np.linspace(0,2,10)
    y = np.linspace(-2,2,10)
    print("10 points:  ", double_integral(fun,x,y))

    x = np.linspace(0,2,50)
    y = np.linspace(-2,2,50)
    print("50 points:  ", double_integral(fun,x,y))

    x = np.linspace(0,2,100)
    y = np.linspace(-2,2,100)
    print("100 points: ", double_integral(fun,x,y))

def main():
    test_double_integral()

if __name__ == "__main__":
    main()
