""" 
-------- EXERCISE 1 - problem 5 ------------
----- FYS-4096 - Computational Physics -----

differentiation.py contains function for the
numerical first derivative of a function

"""

import numpy as np


def first_derivative( function, x, dx ):
    # returns the frist derivative of function
    
    # function: function to derivate
    # x       : location of derivation
    # dx      : accuracy of the derivative
    # order   : order of wanted derivative (1 or 2)
    
    # solves linear system with matrix calculus:
    # f(x+dx) ~ f(x) + dx*f'(x) + h^2/2*f"(x)
    # f(x-dx) ~ f(x) - dx*f'(x) + h^2/2*f"(x)
    # as xv = A^-1 * b | xv = [[f'/dx],[f"/dx^2]]
    
    A = np.matrix([[0.5,-0.5],
                   [1  , 1  ]])
                   
    b = np.matrix([[function(x+dx)-function(x)],
                   [function(x-dx)-function(x)]])
    
    xv = A*b
    
    return xv.item(0)/dx
