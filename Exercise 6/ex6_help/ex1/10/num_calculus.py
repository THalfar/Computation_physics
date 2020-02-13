"""
Various functions for numerical derivation and integration
"""

import numpy as np
import math

"first_derivative: calculates numerical first derivative of funct"
def first_derivative( funct, x, dx ):
	return ( ( funct(x+dx) - funct(x)) / dx )

"test_first_derivative: tests function first_derivative"
def test_first_derivative():
	x = 2
	dx = 0.001
	approximation = first_derivative(function, x, dx) # Approximate numerical solution
	analytic_solution = 12 # Exact analytic solution
	rounded_approximation = np.round(approximation,decimals=0) # Round the approximation
	if rounded_approximation == analytic_solution:
		print (rounded_approximation)
		print ("First derivative correct!")
	else:
		print(rounded_approximation)
		print ("First derivative incorrect!")

"first_derivative: calculates numerical second derivative of funct"
def second_derivative( funct, x, dx ):
	return ( ( funct(x+dx) + funct(x-dx) - (2 * funct(x) ) ) / ( dx**2 ) )

"test_second_derivative: tests function second_derivative"
def test_second_derivative():
	x = 2
	dx = 0.001
	approximation = second_derivative(function, x, dx) # Approximate numerical solution
	analytic_solution = 6 # Exact analytic solution
	rounded_approximation = np.round(approximation,decimals=0) # Round the approximation
	if rounded_approximation == analytic_solution:
		print (rounded_approximation)
		print ("Second derivative correct!")
	else:
		print(rounded_approximation)
		print ("Second derivative incorrect!")		   

"riemann_sum: calculates Riemann sum"
def riemann_sum(x, f):
	delta = x[1]
	i=0
	result = 0
	for i in f:
		result += i*delta
	return result
	
"test_riemann: test function riemann_sum"
def test_riemann():
	x = np.linspace(0,np.pi/2,100) 
	f=np.sin(x) # Integrand function values on uniform grid
	riemann_approx = riemann_sum(x,f)
	rounded_riemann = np.round(riemann_approx,decimals=0)
	analytic_solution = 1
	if rounded_riemann == analytic_solution:
		print (riemann_approx )
		print ("Riemann sum correct!")
	else:
		print(riemann_approx)
		print ("Riemann sum incorrect!")	
		
"trapezoid_sum: calculates integral of x, f using trapezoid sum"
def trapezoid_sum(x, f):
	N=np.size(f)-1
	delta = x[1]
	i=0
	result = 0
	for i in np.arange(N+1): # i gets values from 0 to N
		if (i <= N-1):
			result += (f[i+1]+f[i])*delta
	return result*0.5
	
"test_riemann: test function trapezoid_sum"
def test_trapezoid():
	x = np.linspace(0,np.pi/2,100)
	f = np.sin(x)
	trapezoid_approx = trapezoid_sum(x,f)
	rounded_trapezoid = np.round(trapezoid_approx,decimals=0)
	analytic_solution = 1 # Analytic result
	if rounded_trapezoid == analytic_solution:
		print (trapezoid_approx)
		print ("Trapezoid sum correct!")
	else:
		print(trapezoid_approx)
		print ("Trapezoid sum incorrect!")	
	
	
"trapezoid_sum: calculates integral of x, f using Simpson rule"
def simpson_rule(x, f):
	N = np.size(f) - 1 # N must be one less than the actual size of f to account for indexing from 0
	delta = x[1] # The difference between values of x
	i = 0
	result = 0
	for i in np.arange(N+1): # i gets values from 0 to N
		if (i <= (N/2-1)):
			result += (f[2 * i] + 4 * f[2 * i + 1] + f[2 * i + 2]) # Sum of Simpson rule
	result *= (delta/3) #
	if (np.size(f) % 2) != 0: # If number of intervals is odd, additional term for last slice
			result += (delta/12)*(-f[N-2]+8*f[N-1]+5*f[N])
	return result

"test_simpson(): Function that tests Simpson rule calculation"
def test_simpson():
	x = np.linspace(0,np.pi/2,100)
	f = np.sin(x)
	simpson_approx = simpson_rule(x,f) # Approximate integral with Simpson rule
	rounded_simpson = np.round(simpson_approx,decimals=0)
	analytic_solution = 1 # Analytic result
	if rounded_simpson == analytic_solution:
		print (simpson_approx)
		print ("Simpson sum correct!")
	else:
		print(simpson_approx)
		print ("Simpson sum incorrect!")	

"""monte_carlo_integration(fun,xmin,xmax,blocks,iters): Calculates numerical Monte Carlo integral
of function fun in range xmin, xmax
blocks = number of sections in the area to be integrated
iters = number of iterations per block"""
def monte_carlo_integration(fun,xmin,xmax,blocks,iters):
	block_values=np.zeros((blocks,)) # Ititialise block for values
	L=xmax-xmin # Width of integrable area
	for block in range(blocks):
		for i in range(iters): 
			x = xmin+np.random.rand()*L # Choose random value of x
			block_values[block]+=fun(x) # Add value of function at x into block of values
		block_values[block]/=iters # Divide sum of values by number of iteration
	I = L*np.mean(block_values) 
	dI = L*np.std(block_values)/np.sqrt(blocks)
	return I,dI

def func(x):
	return np.sin(x)


"function(): Example function used for testing integrals"
def function(x):
	return (3 * x ** 2 + 6) 

def main():
	test_first_derivative() 
	test_second_derivative()
	test_riemann()
	test_trapezoid()
	test_simpson()
	
	I,dI=monte_carlo_integration(func,0.,np.pi/2,10,100)
	print("Monte Carlo integral:")
	print(I,'+/-',2*dI)
	
if __name__=="__main__":
	main()
