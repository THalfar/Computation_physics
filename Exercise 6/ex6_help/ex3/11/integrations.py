from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps

def fun4(x,y):
    return (x+y)*exp(-sqrt(x**2+y**2))
    
def main():

    """
    integrate a 2d function fun4 from 0 to 2 and -2 to 2
    """

    x=linspace(0,2,500)
    y=linspace(-2,2,500)
    [X,Y]=meshgrid(x,y)
    F=fun4(X,Y)
    int_dx = simps(F,dx=x[1]-x[0],axis=0)
    int_dxdy = simps(int_dx,dx=y[1]-y[0])
    print('Problem 1.1 = ',int_dxdy)

    x=linspace(0,2,1000)
    y=linspace(-2,2,1000)
    [X,Y]=meshgrid(x,y)
    F=fun4(X,Y)
    int_dx = simps(F,dx=x[1]-x[0],axis=0)
    int_dxdy2 = simps(int_dx,dx=y[1]-y[0])
    print('Problem 1.2 = ',int_dxdy2)

    print('Acc better than ', abs(int_dxdy2-int_dxdy))

if __name__=="__main__":
    main()
