from unittest import TestCase
import numpy as np

from num_calculus.differentiation import first_derivative


class TestNumCalculus(TestCase):        
    def test_first_derivative(self):
        # test function for "first_derivative" using fixed values
        test_function = lambda x: np.sin(x)
        correct_val = 1
        test_x = 0
        test_dx = 0.001
        result = first_derivative(test_function, test_x, test_dx)
        error = abs(correct_val - result)

        print("x: ", test_x, ", dx: ", test_dx, ", 1. derivative: ", result)
        print("Error    is ", error, "\n")

        error_threshold = 1e-3
        self.assertTrue(error < error_threshold)
