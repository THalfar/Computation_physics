from unittest import TestCase
from num_calculus.differentiation import first_derivative


class Test_derivative(TestCase):
    def test_first_derivative():
        print("testing first derivative")

        # basic polynom
        def fun(x):
            return 3 * x ** 2

        print("this should be 12: ", end="")
        print(differentiation.first_derivative(fun, 2, 10e-5))

        # harder polynom
        def fun2(x):
            return 3 * x ** 5 - 2 * x + 4

        print("this should be 238: ", end="")
        print(differentiation.first_derivative(fun2, 2, 10e-5))