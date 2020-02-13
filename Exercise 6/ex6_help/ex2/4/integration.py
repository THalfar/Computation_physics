from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
# from numpy import *

""" FYS-4096 Computational physics """
""" Exercise 2 problem 2 """
""" Roosa Hytönen 255163 """


def func_a1(x):
    return (x**2)*(np.exp(-2*x))


def func_a2(x):
    return np.sinc(x/np.pi)


def func_a3(x):
    return np.exp(np.sin(x**3))


def func_b(x, y):
    return x*(np.exp(-np.sqrt((x**2)+(y**2))))


def integrate_a():
    """ a) Integrates given three functions numerically using the Simpson rule. dx is varied by varying the length of
        vector x containing the values at which each integral is evaluated. The first case is to be evaluated to
        infinity, so let's see what the function looks like to make our calculations easier
    """
    x = np.linspace(0, 100, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, func_a1(x))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'f(x)')
    plt.show()
    """ We note that the first function approaches 0 along with increasing x and thus, evaluation to around x = 20 is 
        more than sufficient
    """
    intervals = [10, 100, 1000, 10000]
    print("a) Numbers of intervals tested: 10, 100, 1000, 10000")
    first_integral = np.zeros(4, )
    second_integral = np.zeros(4, )
    third_integral = np.zeros(4, )
    """ Using a loop to integrate all functions with four different interval numbers, to check for accuracy of the
        estimate (later compared with actual integral value)
    """
    i = 0
    while i < 4:
        x1 = np.linspace(0, 20, intervals[i])
        x2 = np.linspace(0, 1, intervals[i])
        x3 = np.linspace(0, 5, intervals[i])
        first_integral[i] = integrate.simps(func_a1(x1), x1)
        second_integral[i] = integrate.simps(func_a2(x2), x2)
        third_integral[i] = integrate.simps(func_a3(x3), x3)
        i += 1
    print()
    print("First integral")
    print("Actual value: 0.25")
    i = 0
    while i < 4:
        print("{0:.2f}".format(first_integral[i]))
        i += 1
    """ Due to the simple shape of the function, the result is accurate already with 100 sampling intervals
    """
    print()
    print("Second integral")
    print("Actual value: 0.946083")
    i = 0
    while i < 4:
        print("{0:.6f}".format(second_integral[i]))
        i += 1
    """ With 100 intervals the result is already quite accurate 
    """
    print()
    print("Third integral")
    print("Actual value: 6.647272")
    i = 0
    while i < 4:
        print("{0:.6f}".format(third_integral[i]))
        i += 1
    """ Due to the function shape the estimate is accurate only with a larger amount of sampling intervals (10000)
    """


def integrate_b():
    """ Function used for evaluating the double integral in 2b). Using integrate.dblquad to integrate over x and y,
        returns both the value of the integral and an error estimate
    """
    integral_value = integrate.dblquad(func_b, -2, 2, lambda x: 0, lambda x: 2)
    print()
    print("b)")
    print("Actual value: 1.573477")
    print("Computed value: {0:.6f}".format(integral_value[0]))
    print("Error in evaluation:", np.format_float_scientific(integral_value[1],
                                                             unique=False, precision=6))
    """ The function used seems quite accurate
    """


def integrate_c():
    """ Function used for evaluating the double integral in 2c) using lambda-functions. Assuming positions r_a and
        r_b and comparing the estimate to the analytic result
    """
    r_a = (-0.7, 0, 0)
    r_b = (0.7, 0, 0)
    """ Using the information given in the exercise sheet and writing them into lambda-functions
    """
    norm = lambda x: np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    d = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)
    psi = lambda r: np.exp(-norm(r)) / np.sqrt(np.pi)
    integrand = lambda R: abs(psi(np.subtract(R, r_a)))**2 / d(R, r_b)
    analytic_result = lambda r_a, r_b: (1-(1+d(r_a, r_b))*np.exp(-2*d(r_a, r_b)))/d(r_a, r_b)

    """ Evaluation of the integrand in given grid points
    """
    number_of_grid_points = np.linspace(2, 70, 10, dtype=int)
    integral_values = []
    for grid_points in number_of_grid_points:
        pts = np.linspace(-7, 7, grid_points)
        dx = pts[1]-pts[0]
        dy = dx
        dz = dx
        integral = 0
        for x in pts:
            for y in pts:
                for z in pts:
                    r = (x, y, z)
                    integral += integrand(r)*dx*dy*dz
        integral_values.append(integral)
    print()
    print("c)")
    """ Using last value of integral_values as probably the best estimate
    """
    print("Analytic result: {:.6f}".format(analytic_result(r_a, r_b)))
    print("Numerical estimate: {:.6f}".format(integral_values[-1]))
    print("Deviation: {:.6f}".format(analytic_result(r_a, r_b)-integral_values[-1]))
    """ Computing is quite slow but eventually the result is obtained for one point
    """


def main():
    integrate_a()
    integrate_b()
    integrate_c()


if __name__ == "__main__":
    main()
