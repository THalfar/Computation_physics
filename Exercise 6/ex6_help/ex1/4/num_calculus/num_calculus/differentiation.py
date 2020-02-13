"""
This file contains the code for numerically approximating the
first derivative of a function.

This file is part of the course "Computational Physics"

Juha Teuho
"""

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