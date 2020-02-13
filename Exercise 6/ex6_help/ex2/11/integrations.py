from numpy import *
from num_calculus import *
from matplotlib.pyplot import *
from scipy.integrate import simps

def fun1(x):
    return x**2*exp(-2*x)

def fun2(x):
    return sinc(x/pi)

def fun3(x):
    return exp(sin(x**3))

def fun4(x,y):
    return x*exp(-(x**2+y**2))

def fun5(x,y,z,r_a,r_b):
    f=exp(-2*sqrt((x-r_a[0])**2+(y-r_a[1])**2+(z-r_a[2])**2))/pi
    f/=sqrt((x-r_b[0])**2+(y-r_b[1])**2+(z-r_b[2])**2)
    return f

def fun5_exact_int(r_a,r_b):
    R=sqrt(sum((r_a-r_b)**2))
    return (1-(1+R)*exp(-2*R))/R

def own_sinc(x):
    if isscalar(x):
       x=array(x)
    f=zeros((len(x),))
    for i in range(len(x)):
        if (x[i]<1e-8):
            f[i]=1.-x[i]**2/6
        else:
            f[i]=sin(x[i])/x[i]
    return f

def main():
    """
    integrate fun1 from 0 to infty
    - the integrand converges towards zero exponentially
    - exact answer is 0.25
    """
    x=linspace(0,20,1000)
    I1=trapezoid(x,fun1)
    print('2a (1)   = ', I1)

    """
    integrate fun2 from 0 to 1
    - the integrand is a sinc function
    - that is as sin(x)/x approaches 1 as x->0
    """
    x=linspace(0,1,1000)
    I2=trapezoid(x,fun2)
    print('2a (2)   = ',I2)
    print('2a (2.1) = ',trapezoid(x,own_sinc))

    """
    integrate fun3 from 0 to 5
    - the integrand is a rapidly oscillating function when x>1.7
    - needs a fine grid in that region 
    - a change of variable could also be good 
    """
    x=linspace(0,5,1000)
    I3=trapezoid(x,fun3)
    print('2a (3)   = ',I3)

    """
    integrate a 2d function fun4 from 0 to 2 and -2 to 2
    """
    x=linspace(0,2,1000)
    y=linspace(-2,2,1000)
    [X,Y]=meshgrid(x,y)
    F=fun4(X,Y)
    int_dx = simps(F,dx=x[1]-x[0],axis=0)
    int_dxdy = simps(int_dx,dx=y[1]-y[0])
    print('2b       = ',int_dxdy)

    """
    integrate a 3d function fun5 over a volume
    """
    x=linspace(-7,7,300)
    y=linspace(-7,7,300)
    z=linspace(-7,7,300)
    [X,Y,Z]=meshgrid(x,y,z)
    r_a=array([-0.7,0,0])
    r_b=array([0.7,0,0])
    F=fun5(X,Y,Z,r_a,r_b)
    int_dx = simps(F,dx=x[1]-x[0],axis=0)
    int_dxdy = simps(int_dx,dx=y[1]-y[0],axis=0)
    int_dxdydz = simps(int_dxdy,dx=z[1]-z[0])
    print('2c       = ',int_dxdydz)
    print('2c exact = ',fun5_exact_int(r_a,r_b))

if __name__=="__main__":
    main()
