from numpy import *
from scipy import *
from scipy.integrate import quad, nquad, tplquad, simps
from num_calculus import   rect_int_3d, simpson_integration

"""
In this module all the printing in LaTeX format
"""
"""
Problem 2a
"""

"""
Defining the integrands and takinf definite intergrals
"""
def integrand_a1(r):
    return exp(-2*r)*r**2

int_a1,err_a1= quad(integrand_a1,0,inf) #quad is ok with improper 1d integral
print('\int_{0}^{\infty} \exp(-2r)r² = ',int_a1,'+/-',err_a1)


def integrand_a2(x):
    return sin(x)/x

int_a2,err_a2= quad(integrand_a2,0,1) #quad method is ok with sinc function
print('\int_{0}^{1} \frac{\sin x}{x} = ',int_a2,'+/-',err_a2)


def integrand_a3(x):
    return exp(sin(x**3))

int_a3,err_a3= quad(integrand_a3,0,5)
print('\int_{0}^{5} \exp(\sin(x³)) = ',int_a3,'+/-',err_a3)

"""
Problem 2b
"""
def integrand_b(x,y):
    return x*exp(-sqrt(x*x+y*y))
x = linspace(0, 2, 30)
y = linspace(-2, 2, 30)
[X,Y] = meshgrid(x,y)

int_b,err_b= nquad(integrand_b,[[0,2],[-2,2]]) #nquad is recursive and it works badly with large areas but here we have a quite simple double integral
print('\int_{0}^{2} \int_{-2}^{2} x*exp(-\sqrt{*x+y*y}) = ',int_b,'+/-',err_b)
"""
quad and nquad methods are also good to estimate the error, so we can do it without varying grid sizes
"""

"""
Problem 2c
"""


def j(x,y,z,a,b):
    return (exp(-2*sqrt((x-a[0])**2+(y-a[1])**2+(z-a[2])**2)))/(pi*((x-b[0])**2+(x-b[1])**2+(x-b[2])**2))
    """
    defining of Psi function from Exercise 2
    """
xa = -0.7
xb = 0.7
x = linspace(-15, 15, 100)
y = linspace(-15, 15, 100)
z = linspace(-15, 15, 100)
[X,Y,Z] = meshgrid(x,y,z)

"""
5 given points to estimate how importand is the choise of approximate infinite volume
"""
j1=j(X,Y,Z,[xa,0,0],[xb,0,0])
j2=j(X,Y,Z,[1,2,3],[4,5,6])
j3=j(X,Y,Z,[4,5,6],[2,3,4])
j4=j(X,Y,Z,[4,2,1],[2,3,4])
j5=j(X,Y,Z,[3,2,1],[1,3,4])


int_c1 = simps(simps(simps(j1, dx=x[1]-x[0],axis=0),  dx=y[1]-y[0],axis=0), dx=z[1]-z[0])

int_c2 = simps(simps(simps(j2, dx=x[1]-x[0],axis=0),  dx=y[1]-y[0],axis=0), dx=z[1]-z[0])

int_c3 = simps(simps(simps(j3, dx=x[1]-x[0],axis=0),  dx=y[1]-y[0],axis=0), dx=z[1]-z[0])

int_c4 = simps(simps(simps(j4, dx=x[1]-x[0],axis=0),  dx=y[1]-y[0],axis=0), dx=z[1]-z[0])

int_c5 = simps(simps(simps(j5, dx=x[1]-x[0],axis=0),  dx=y[1]-y[0],axis=0), dx=z[1]-z[0])


print('j_1 = ',int_c1)
print('j_2 = ',int_c2)
print('j_3 = ',int_c3)
print('j_4 = ',int_c4)
print('j_5 = ',int_c5)

"""
Analytical value (see Ex.(2) notes)
"""
def j_an(R):
    return (1-(1+R)*exp(-2*R))/R

"""
Comparing with analytical value of j
"""
print('Error in the 1st point', abs(int_c1-j_an(sqrt(sum(subtract([xa,0,0],[xb,0,0])**2)))))
print('Error in the 2nd point', abs(int_c1-j_an(sqrt(sum(subtract([1,2,3],[4,5,6])**2)))))
print('Error in the 3rd point', abs(int_c1-j_an(sqrt(sum(subtract([4,5,6],[2,3,4])**2)))))
print('Error in the 4th point', abs(int_c1-j_an(sqrt(sum(subtract([4,2,1],[2,3,4])**2)))))
print('Error in the 5th point', abs(int_c1-j_an(sqrt(sum(subtract([3,2,1],[1,3,4])**2)))))
print('Such divergence means that we take too small integration volume. When we take a small volume of integration, we have to take coordinates of r_a and r_b as small as possible to get some accuracy in real calculation, also it is necessary to monitor the density of points, but that will liad to increase in calculation time.')
