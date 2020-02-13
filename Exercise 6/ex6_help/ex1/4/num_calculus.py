""" 
This file contains calculus-functions for
exercise 1 on course 'Computational Physics'

Juha Teuho
"""

import numpy as np

def first_derivative( fun, x, dx ):
	"""
	This function approximates the value of the first
	derivative of a function by using the symmetric
	difference quotient.

	params:
	fun: 	The function that the derivative 
			is approximated for
	x:		Variable
	dx:		A small change in x		

	return:		
	The approximation for the first 
	derivative of a function
	"""
	return (fun(x + dx) - fun(x - dx)) / (2 * dx)


def second_derivative(fun, x, dx):
    """
    This function approximates the value of the second
    derivative of a function using the second-order
    central difference method.

    params:
	fun: 	The function that the derivative
			is approximated for
	x:		Variable
	dx:		A small change in x

	return:
	The approximation for the second
	derivative of a function
    """

    return (fun(x + dx) + fun(x - dx) - 2 * fun(x)) / (dx ** 2)

def riemann_sum( x, fun ):
    """
    This function approximates the value integral of a
    function at a given interval using Riemann sum.

    params:
    x:      Interval as a numpy-array.
    fun: 	The function that the integral is approximated for

    return:
    The approximation for the integral of the function.
    """


    # comments
    sum = 0
    dx = x[1]-x[0]
    if x.size > 1:
            for i in range(0,x.size-1):
                    sum += fun(x[i])*dx
    else:
            print("Can't perform Riemann's sum with less than 2 points.")
    return sum

def trapezoidal( x, fun ):
        """
        This function approximates the value integral of a
        function at a given interval using the trapezoidal rule.

        params:
        x:      Interval as a numpy-array.
        fun: 	The function that the integral is approximated for

        return:
        The approximation for the integral of the function.
        """


        # comments
        sum = 0
        if x.size > 1:
                dx = x[1] - x[0]
                for i in range(0,x.size-2):
                        sum += 1/2*(fun(x[i])+fun(x[i+1]))*dx
        else:
                print("Can't perform trapezoidal sum with less than 2 points.")
        return sum

def simpson( x, fun ):
        """
        This function approximates the value integral of a
        function at a given interval using the simpson's rule.

        params:
        x:      Interval as a numpy-array.
        fun: 	The function that the integral is approximated for

        return:
        The approximation for the integral of the function.
        """

        # comments
        sum = 0

        if x.size < 3:
                print("Can't perform Simpson sum with less than 3 points.")
                return

        dx = x[1]-x[0]

        if x.size % 2 == 0:
                for i in range(0,x.size//2-2):
                        sum += dx / 3 * (fun(x[2 * i]) + 4 * fun(x[2 * i + 1]) + fun(x[2 * i + 2]))
                dI = dx / 12 * (-fun(x[x.size - 3]) + 8 * fun(x[x.size - 2]) + 5 * fun(x[x.size-1]))
                return sum + dI
        else:
                for i in range(0,x.size//2-1):
                        sum += dx / 3 * (fun(x[2 * i]) + 4 * fun(x[2 * i + 1]) + fun(x[2 * i + 2]))
                return sum


def monte_carlo_integration( fun, xmin, xmax, blocks, iters):
        """
        This function approximates the value integral of a
        function at a given interval using monte carlo intergration

        params:
        fun:    The function that the integral is approximated for
        xmin:   Lower limit of the interval
        xmax:   Upper limit of the interval
        blocks:	Number of blocks
        iters:	Number of iterations

        return:
        The approximation for the integral of the function.
        """

        block_values = np.zeros((blocks,))
        L=xmax-xmin
        for block in range(blocks):
                for i in range(iters):
                        x = xmin+np.random.rand()*L
                        block_values[block]+=fun(x)
                block_values[block]/=iters
        I = L*np.mean(block_values)
        dI = L*np.std(block_values)/np.sqrt(blocks)
        return I,dI

"""
Define test function and it's derivative, second derivative and integral
"""
def test_function(x):
        return 3*x**3

def der_test_function(x):
        return 9*x**2

def second_der_test_function(x):
        return 18*x

def int_test_function(xmin, xmax):
		return 3/4*xmax**4-3/4*xmin**4

def test_first_derivative():
        """
        Test function for first_derivative
        If the absolute error is less than the maximum error,
        test is OK - otherwise it fails.
        """

        x = 2.0
        dx = 0.0001
        max_error = 0.001
        numerical = first_derivative(test_function, x, dx)
        analytical = der_test_function(x)

        if abs(numerical-analytical) < max_error:
                print("first derivative test OK")
        else:
                print("first derivative test failed")

def test_second_derivative():
        """
        Test function for second_derivative
        If the absolute error is less than the maximum error,
        test is OK - otherwise it fails.
        """

        x = 2.0
        dx = 0.0001
        max_error = 0.001
        numerical = second_derivative(test_function, x, dx)
        analytical = second_der_test_function(x)

        if abs(numerical-analytical) < max_error:
                print("second derivative test OK")
        else:
                print("second derivative test failed")
        
def test_riemann_sum():
        """
        Test function for riemann_sum
        If the absolute error is less than the maximum error,
        test is OK - otherwise it fails.
        """
        xmin = 0
        xmax = 1
        intervals = 10000
        max_error = 0.001
        x = np.linspace(xmin,xmax,num=intervals+1)

        analytical = int_test_function(xmin, xmax)
        numerical = riemann_sum(x, test_function)

        if abs(numerical-analytical) < max_error:
                print("riemann sum test OK")
        else:
                print("riemann sum test failed")

def test_trapezoidal():
        """
        Test function for trapezoidal
        If the absolute error is less than the maximum error,
        test is OK - otherwise it fails.
        """

        xmin = 0
        xmax = 1
        intervals = 10000
        max_error = 0.001
        x = np.linspace(xmin,xmax,num=intervals+1)

        analytical = int_test_function(xmin, xmax)
        numerical = trapezoidal(x, test_function)

        if abs(numerical - analytical) < max_error:
                print("trapezoidal test OK")
        else:
                print("trapezoidal test failed")

def test_simpson():
        """
        Test function for simpson
        If the absolute error is less than the maximum error,
        test is OK - otherwise it fails.
        """

        xmin = 0
        xmax = 1
        intervals = 10000
        max_error = 0.001
        x = np.linspace(xmin,xmax,num=intervals+1)

        analytical = int_test_function(xmin, xmax)
        numerical = simpson(x, test_function)

        if abs(numerical - analytical) < max_error:
                print("simpson test OK")
        else:
                print("simpson test failed")

def test_monte_carlo_integration():
        """
        Test function for monte carlo integration
        If the absolute error is less than the maximum error,
        test is OK - otherwise it fails.
        """
        xmin = 0
        xmax = 1
        blocks = 10
        iterations = 10000
        max_error = 0.01

        analytical = int_test_function(xmin, xmax)
        numerical, dI = monte_carlo_integration(test_function,xmin,xmax,blocks,iterations)

        if abs(numerical - analytical) < max_error:
                print("monte carlo test OK")
        else:
                print("monte carlo test failed")

def main():
        # perform the tests
        test_first_derivative()
        test_second_derivative()
        test_riemann_sum()
        test_trapezoidal()
        test_simpson()
        test_monte_carlo_integration()

if __name__=="__main__":main()
