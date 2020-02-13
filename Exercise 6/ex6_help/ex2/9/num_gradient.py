""" CompPhys exercise 2 problem 3
 contains functions for N-dimensional numerical gradient and gradient descent"""
import numpy as np

def gradient(x0, fun, eps=0.01):
    """
    Numerical gradient of N-dimensional function fun
    :param x0: Coordinate to calculate gradient at, array
    :param fun: Function handle, must accept numpy array as argument
    :param eps: Gradient epsilon, default 0.01
    :return: Numerical gradient array
    """
    grad = np.full_like(x0,0) # Initialization
    for i in range(np.size(x0)):
        # eps_array is zeroed for all but the axis being differentiated at the iteration
        eps_array = np.zeros_like(x0)
        eps_array[i] = eps
        # Center difference method generalized to N-dimensions, takes N-loops to complete
        grad[i] = (fun(x0+eps_array)-fun(x0-eps_array))/(2*eps)
    return grad

def gradient_descent(x0, fun, eps=0.01, tol=1e-15, max_steps=10000, descent=True):
    """
    Gradient descent (ascent), returns coordinate of closest minima (maxima)
    :param x0: Initial coordinate
    :param fun: Function handle, must accept numpy array as argument
    :param eps: Gradient epsilon and descent epsilon value, default 0.01
    :param max_steps: Maximum amount of steps before aborting
    :param descent: True for minima, false for maxima
    :return: Coordinate of closest minimum or maximum, NaN if max steps reached
    """
    steps = 0
    # Ascent/descent handled by multiplying gradient with -1
    mul = 1
    if descent == False:
        mul = -1
    while steps < max_steps:
        x1 = x0 - mul*eps*gradient(x0, fun, eps)
        delta = np.abs(np.linalg.norm(x1-x0))
        if delta < tol:
            return x1
        x0 = x1
        steps += 1
    print("Maximum steps taken, returning NaN")
    return np.nan

def test_gradient():
    # Lazy gradient test function
    # Test cases x+y+z, x+xy+z^3
    fun = lambda x0: x0[0]+x0[1]+x0[2]
    x0 = np.array(([1],[1],[1]),np.float)
    grad = gradient(x0, fun)
    # Tests if all elements within default tolerance
    if not np.allclose(np.array(([1.],[1.],[1.])), grad):
        print("Gradient test failed")
        return
    fun = lambda x0: x0[0] + x0[1]*x0[0] + x0[2]**3
    x0 = np.array(([1], [1], [1]), np.float)
    grad = gradient(x0, fun, eps=1e-9)
    # Tests if all elements within default tolerance
    if not np.allclose(np.array(([2.], [1.], [3.])), grad):
        print("Gradient test failed")
        return
    print("Gradient test successful")

def test_gradient_descent():
    fun = lambda x0: (1-x0[0])**2-x0[1]**2
    x0 = np.array(([1],[1]),np.float)
    grad = gradient_descent(x0, fun, descent=False)
    if not np.allclose(np.array(([1.], [0.])), grad):
        print("Gradient descent test failed")
        return
    print("Gradient descent test successful")

def main():
    test_gradient()
    test_gradient_descent()

if __name__=="__main__":
    main()