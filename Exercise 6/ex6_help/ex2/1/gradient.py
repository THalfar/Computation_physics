""" module to calculate N-dimensional numerical gradient of function."""
import numpy as np
import copy

def gradient(fun, h, point):
    """
    function for numerical N-dimensianal gradient
    :param fun: function which takes variables as list
    :param h: also known as dx or step size
    :param point: where you calculate gradient
    :return: array containing gradient
    """
    # create empty array
    gradient = np.zeros(len(point))
    # looping through N dimensions
    for i in range(len(point)):
        # copy of points list to allow modification with h without losing origal point
        d_point = copy.deepcopy(point)
        d_point[i] = d_point[i]+h
        # calculating gradient
        gradient[i] = (fun(d_point) - fun(point)) / h
    return gradient

def fun1(x):
    # 1d test function
    return x[0]**2

def fun2(x):
    # 2d test function
    return x[0]*x[1]**2+1

def test_gradient():
    """
    testing gradient and printing results compared to real values
    """
    print("this should be [2]: ", gradient(fun1, 10e-10, [1]))
    print("this should be [1 2]: ", gradient(fun2, 10e-10, [1,1]))

def main():
    test_gradient()

if __name__ == "__main__":
    main()