import numpy as np
from matplotlib.pyplot import *

def eval_derivative(func,x,h):
    return (func(x+h)-func(x-h))/2/h

def eval_partial_derivative(func,x,dx,dim):
    h=dx[dim]
    dx=0.0*dx
    dx[dim]=h
    return (func(x+dx)-func(x-dx))/2/h

def gradient(func,x,h):
    if np.isscalar(x):
        grad=eval_derivative(func,x,h)
    else:
        grad=np.zeros((len(x),))
        for i in range(len(x)):
            grad[i]=eval_partial_derivative(func,x,h,i)
    return grad

def steepest_fit(func,x,tol,a_coef):
    dtol=dtol_old=100.
    i=0
    X=1.0*x
    h = 0.01*np.ones(np.shape(x))
    ngrad_old=1000.
    while dtol>tol and i<1000: 
        grad=gradient(func,X,h)
        ngrad=np.sqrt(np.sum(grad**2))
        dX=-a_coef*grad/(ngrad+1.)
        X = X+dX
        dtol=np.sqrt(np.sum(dX**2))
        i+=1
    return X,i

def testing_fun(x):
    return np.sum(x**2)


def main():

    x = np.ones((10,))
    tol = 1.0e-6
    a_coef = 0.5
    X,i=steepest_fit(testing_fun,x,tol,a_coef)
    print('Number of iterations',i)
    print('Minimum at ',X)


if __name__=="__main__":
    main()

