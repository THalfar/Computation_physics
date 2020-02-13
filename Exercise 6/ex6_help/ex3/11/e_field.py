from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps

def de_field_1d_rod(x,r0,const=1.0):
    """
    dE = lambda * (4 pi epsilon0)^(-1) * dx/r^2 \hat{r},
    
    where lambda='line charge density' and hat{r} is unit vector
    from pointing from dx to desired point r0.
    
    Here we use const = lambda * (4 pi epsilon0)^(-1) = 1.
    """
    r_rod=0.0*r0
    r_rod[0]=x
    r = (r0-r_rod)
    norm_r2 = sum(r**2)
    hat_r = r/sqrt(norm_r2)
    return const/norm_r2*hat_r

def calc_rod_e_field(x_rod,r0,const=1.0):
    """
    Calculates the net electric field due to a charged rod
    located on x-axis. Simpson rule used for integration.
    x_rod defines both the rod length and the integration
    spacing. Uniform grid assumed. Integrand is given by
    function de_field_1d_rod.
    """
    dE=zeros((len(x_rod),len(r0)))
    E=zeros((len(r0),))
    dx=x_rod[1]-x_rod[0]
    for i in range(len(x_rod)):
        dE[i,:]=de_field_1d_rod(x_rod[i],r0)
    for i in range(len(r0)):
        E[i]=simps(dE[:,i],dx=dx)
    return E

def test_on_x_axis(L,d,const=1):
    """
    Analytical result for a point distance 
    d away from the edge the on the positive side.
    For example at r=(x0,y0)=(L/2+d,0), when the center
    of the rod is at the origing.
    """
    return const*(1.0/d-1.0/(L+d)) 


def main():
    L = 2.0
    x=linspace(-L/2,L/2,100)
    d=2.0
    r_test=array([L/2+d,0])
    print('Comparing to analytical results')
    comput = calc_rod_e_field(x,r_test)
    exact = test_on_x_axis(L,d)
    E_diff = abs(comput[0]-exact)
    print('  Simpson   : ', comput)
    print('  Analytical: ', exact)
    print('  difference: ', E_diff)
    if (E_diff<1e-4):
        print('  accuracy  : ', 'OK')
    else:
        print('  accuracy  : ', 'NOT OK')
        exit()
    
    xmax=L/2+d
    dim=20
    x0=linspace(-xmax,xmax,dim)
    y0=1.0*x0
    Ex=zeros((dim,dim))
    Ey=zeros((dim,dim))
    for i in range(len(x0)):
        for j in range(len(y0)):
            r0=array([x0[i],y0[j]])
            E=calc_rod_e_field(x,r0)
            #E=E/sqrt(sum(E**2)) # normalized vector field
            Ex[i,j]=E[0]
            Ey[i,j]=E[1]

    [X0,Y0] = meshgrid(x0,y0)
    quiver(transpose(X0),transpose(Y0),Ex,Ey)
    #quiver(X0,Y0,Ex,Ey)
    plot(x,zeros(shape=shape(x)),'b',linewidth=2)
    show()

if __name__=="__main__":
    main()
    
