from numpy import *
from scipy import *

from num_calculus import test_fun2
"""
Gradient itself
"""
def num_gradient(f,x,dx):
    temp = f(x)
    grad = zeros((len(x),))
    h = zeros((len(x),))    
    for i in range(len(x)):
        h[i]=dx[i]
        grad[i]=(f(x+h)-f(x-h))/2/dx[i]
        h[i]=0.0
    return grad


"""
Gradient descent, see Eq.(31-32) from Week 2 FYS-4096 lecture notes
"""
def gradient_descent(f,x):
    if isscalar(x):
        x=array([x])
    a = 0.01
    i = 0 #added later
    dx = 0.0*x+0.001
    dX = a/(sqrt(sum(num_gradient(f,x,dx)**2))+1)*num_gradient(f,x,dx)
    while sqrt(sum(dX**2))>1.0e-4 and i<1000:
        grad = num_gradient(f,x,dx)
        gamma=a/(sqrt(sum(grad**2))+1)
        dX=gamma*grad
        x=x-dX
	i+=1 #added later
    return x


        


""" Analytical test function definitions """
def test_fun_another_one(x):
    """
    this is the function F(x,y)=x exp(-sqrt(x²+y²))
    Gradient of this function in (1,2) is (0.0590807,-0.0955945)
    """
    return x[0]*exp(-sqrt(x[0]**2+x[1]**2))

def test_fun_grad():
    return array([0.0590807,-0.0955945])

def test_fun_grad():
    return array([0.0590807,-0.0955945])

""" Test routines for unit testing """
def test_gradient(tolerance=1.0e-5):
     """ Test routine for gradient of f. The value of gradient of given function in (1,2) is (0.0590807,-0.0955945) """
     x = array([1,2])
     dx = array([0.001,0.001])
     gr_estimate = num_gradient(test_fun_another_one,x,dx)
     gr_exact = test_fun_grad()
     err = abs(sum(gr_estimate[:]-gr_exact[:]))
     working = False
     if (err<tolerance):
         print('Gradient is OK')
         working = True
     else:
         print('Gradient is NOT ok!!')
     return working

def test_gradient_descent(tolerance=1.0e-2):
     """ Test routine for gradient of f. The value of gradient of given function in (1,2) is (0.0590807,-0.0955945) """
     x = 3
     grd_estimate =  gradient_descent(test_fun2,3)
     grd_exact =array([3*pi/2]) #we already know that extremum if sin is in poin of 3pi/2 that near the starting point of 3
     err = abs(sum(grd_estimate[:]-grd_exact[:]))
     working = False
     if (err<tolerance):
         print('Gradient descent is OK')
         working = True
     else:
         print('Gradient descent is NOT ok!!')
     return working


#print(gradient_descent(sin,3),3*pi/)) #not needed
""" Tests performed in main """
def main():
    """ Performing all the tests related to this module """
    test_gradient()
    test_gradient_descent()

if __name__=="__main__":
    main()
