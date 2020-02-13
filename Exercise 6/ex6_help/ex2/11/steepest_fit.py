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
    counter = 0
    ngrad_old=1000.
    while dtol>tol and i<1000: 
        grad=gradient(func,X,h)
        ngrad=np.sqrt(np.sum(grad**2))
        dX=-a_coef*grad/(ngrad+1.)
        X = X+dX
        dtol=np.sqrt(np.sum(dX**2))
        i+=1
    return X,i

def fit_fun(a,x):
    return a[0]*np.exp(-a[1]*x)

def data_on_grid():
    x = np.linspace(0,5,10)
    y = 5.*np.exp(-3.*x)+0.01*(0.5-np.random.random(np.shape(x)))
    return x,y

def optim_fun(a):
    f = fit_fun(a,x_data)
    return np.sqrt(np.sum((f-y_data)**2))

def main():
    rcParams['legend.handlelength'] = 2
    rcParams['legend.numpoints'] = 1
    rcParams['text.usetex'] = True
    rcParams['font.size'] = 18

    global x_data
    global y_data
    x_data, y_data = data_on_grid()
    x = np.array([2.,1.])
    tol = 1.0e-6
    a_coef = 0.05
    X,i=steepest_fit(optim_fun,x,tol,a_coef)
    print(X,i)

    r=np.linspace(0,10,1000)
    plot(x_data,y_data,'o',label='data')
    plot(r,fit_fun(x,r),'--',label='start')
    plot(r,fit_fun(X,r),label='final')
    legend(loc=0)
    show()

if __name__=="__main__":
    main()

