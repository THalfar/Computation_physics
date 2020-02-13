"""
Linear interpolation in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019

Modified by Roman Goncharov by January 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from num_calculus import test_fun2


"""
Basis functions l1 and l2 are here

"""
def l1(t):
    """ 
    It just the first basis linear interpolation function
    See, e.g., Week 2 FYS-4096 lecture notes (Eq.(1)).
    """
    return 1-t
def l2(t):
    """ 
    The second basis linear interpolation function
    See, e.g.,  Week 2 FYS-4096 lecture notes. 
    """
    return t

class linear_interp:
    """
    Check of dimensionality, adding attributes, it should be 1-3D
    So here we fix function and argument dimensionality and define the denominators for t (see Week 2 FYS-4096 lecture notes)
    """

    def __init__(self,*args,**kwargs):
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
            self.hz=np.diff(self.z)
        else:
            print('Either dims is missing or specific dims is not available')

    """
    Evaluation in different dims, the rules the same as in  Week 2 FYS-4096 lecture notes.
    """
    def eval1d(self,x):
        if np.isscalar(x): # checking if the given x is the set of points
            x=np.array([x]) 
        N=len(self.x)-1
        f=np.zeros((len(x),))
        ii=0
        for val in x:
            i=np.where(self.x<=val)[0][-1] #indexing
            """
            There was i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)
            but in our case we can just use the simpler one
            this line can already give us a desired index
            """
            if i==N: # for the last point
                f[ii]=self.f[i]
            else:
                t=(val-self.x[i])/self.hx[i] 
                f[ii]=self.f[i]*l1(t)+self.f[i+1]*l2(t) #Eq.(1) from Week 2 lecture slides exactly
            ii+=1
        return f

    """
    For 2D and 3D evaluation is the same, just step-by-step through one axis 
    having the values on other ones fixed, so we got nested loops.
    We define matrices A consisting function values in neighbouring points. 
    p are vectors of l, there're not defined in lectures
    but it is convinient to use it here
    """
    def eval2d(self,x,y):
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        f=np.zeros((len(x),len(y)))
        A=np.zeros((2,2))
        ii=0
        for valx in x:
            i=np.where(self.x<=valx)[0][-1]
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.where(self.y<=valy)[0][-1]
                if (j==Ny):
                    j-=1
                tx = (valx-self.x[i])/self.hx[i]
                ty = (valy-self.y[j])/self.hy[j]
                ptx = np.array([l1(tx),l2(tx)])
                pty = np.array([l1(ty),l2(ty)])
                A[0,:]=np.array([self.f[i,j],self.f[i,j+1]])
                A[1,:]=np.array([self.f[i+1,j],self.f[i+1,j+1]])
                f[ii,jj]=np.dot(ptx,np.dot(A,pty)) #Eq.(2) from Week 2 lecture slides exactly
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        if np.isscalar(z):
            z=np.array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=np.zeros((len(x),len(y),len(z)))
        A=np.zeros((2,2))
        B=np.zeros((2,2))
        ii=0
        for valx in x:
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                for valz in z:
                    k=np.floor(np.where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1
                    tx = (valx-self.x[i])/self.hx[i]
                    ty = (valy-self.y[j])/self.hy[j]
                    tz = (valz-self.z[k])/self.hz[k]
                    ptx = np.array([l1(tx),l2(tx)])
                    pty = np.array([l1(ty),l2(ty)])
                    ptz = np.array([l1(tz),l2(tz)])
                    B[0,:]=np.array([self.f[i,j,k],self.f[i,j,k+1]])
                    B[1,:]=np.array([self.f[i+1,j,k],self.f[i+1,j,k+1]])
                    A[:,0]=np.dot(B,ptz)
                    B[0,:]=np.array([self.f[i,j+1,k],self.f[i,j+1,k+1]])
                    B[1,:]=np.array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1]])
                    A[:,1]=np.dot(B,ptz)
                    f[ii,jj,kk]=np.dot(ptx,np.dot(A,pty)) #Eq.(3) from Week 2 lecture slides exactly
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
    
# end class linear interp

""" Testing """
def test_linear_interp_1D(tolerance=1.0e-2): #since we want to show interpolation with not so many given points, there should be small tolerance 
    """ Test routine for 1D interpolation"""
    x=np.linspace(0.,2.*np.pi,10) #given grid
    y=np.sin(x) #given function
    xx=3 #given point
    lin1d_f = linear_interp(x=x,f=y,dims=1) #since the use classes we should do it first
    f_est=lin1d_f.eval1d(xx) #and then we can apply the interpolation function
    f_exact = test_fun2(xx)
    err = np.abs(f_est-f_exact)
    working = False
    if (err<tolerance):
        print('1D linear interpolation is OK')
        working = True
    else:
        print('1D linear interpolation is NOT ok!!')
    return working
"""
For 2D and 3D testing exactly the same
"""

def test_linear_interp_2D(tolerance=1.0e-1):
    """ Test routine for 2D linear interpolation"""
    xx=1.5
    yy=-1.5
    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y)
    Z = X*np.exp(-1.0*(X*X+Y*Y))
    lin2d_f=linear_interp(x=x,y=y,f=Z,dims=2)
   
    f_est=lin2d_f.eval2d(xx,yy)
    f_exact = test_fun3(xx,yy)
    err = np.abs(f_est-f_exact)
    working = False
    if (err<tolerance):
        print('2D linear interpolation is OK')
        working = True
    else:
        print('2D linear interpolation is NOT ok!!')
    return working

def test_linear_interp_3D(tolerance=1.0e-2):
    """ Test routine for 3D interpolation"""
    xx=1.5
    yy=1
    zz=2

    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,10)
    z=np.linspace(0.0,3.0,10)
    
    X,Y,Z = np.meshgrid(x,y,z)
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= np.meshgrid(x,y)
    lin3d_f=linear_interp(x=x,y=y,z=z,f=F,dims=3)

    f_est=lin3d_f.eval3d(xx,yy,zz)
    f_exact = test_fun4(xx,yy,zz)
    err = np.abs(f_est-f_exact)
    working = False
    if (err<tolerance):
        print('3D linear interpolation is OK')
        working = True
    else:
        print('3D  interpolation is NOT ok!!')
    return working

""" Analytical test function definitions """
def test_fun3(x,y):
    """
    x*exp{-(x²+y²)} is used for the 2D interpolation test.
    Should give \approx 135 in point (1.5,-1.5) for the result.
    """
    return x*np.exp(-1.0*(x*x+y*y))

def test_fun4(x,y,z):
    """
    (x+y+z)*exp{-(x²+y²+z²)} is used for the 3D interpolation test.
    Should give \approx 6336.47 in point (1.5,1,2) for the result.
    """
    return (x+y+z)*np.exp(-1.0*(x*x+y*y+z*z))

""" Tests and plots performed in main """
def main():        
    """ Firstly, performing all the tests related to this module """
    test_linear_interp_1D()
    test_linear_interp_2D()
    test_linear_interp_3D()
    """ Secondly, performing plotting to this module """ 
   # 1d example
    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)
    
 
    x=np.linspace(0.,2.*np.pi,10)
    y=np.sin(x)
    lin1d=linear_interp(x=x,f=y,dims=1)
    xx=np.linspace(0.,2.*np.pi,100)
    ax1d.plot(xx,lin1d.eval1d(xx),label='Interpolated')
    ax1d.plot(xx,np.sin(xx),'r--',label='Actual function values')
    ax1d.plot(x,y,'o')
    ax1d.set_title('1D linear interpolation of $\sin (x)$')
    ax1d.legend(loc=0)
    fig1d.savefig('exercise2_1d_lin.pdf',dpi=200)

    # 2d example
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(221, projection='3d')
    ax2d2 = fig2d.add_subplot(222, projection='3d')
    ax2d3 = fig2d.add_subplot(223)
    ax2d4 = fig2d.add_subplot(224)

    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y)
    Z = X*np.exp(-1.0*(X*X+Y*Y))
    ax2d.plot_wireframe(X,Y,Z) #3d plotting
    ax2d3.pcolor(X,Y,Z) #2D projection colour plotting

    lin2d=linear_interp(x=x,y=y,f=Z,dims=2)
    x=np.linspace(-2.0,2.0,51)#more points
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y)
    Z = lin2d.eval2d(x,y)
     
    ax2d2.plot_wireframe(X,Y,Z)
    ax2d4.pcolor(X,Y,Z)
    ax2d.set_title('Small grid')
    ax2d2.set_title('Large grid')
    fig2d.suptitle('2D linear interpolation of $xe^{-(x²+y²)}$')
    fig2d.savefig('exercise2_2d_lin.pdf',dpi=200)
    # 3d example
    """
    This function exits in 4D space, so we can only plot the projection
    """
    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,10)
    z=np.linspace(0.0,3.0,10)
    X,Y,Z = np.meshgrid(x,y,z)
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= np.meshgrid(x,y)
    fig3d=plt.figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)-1)])
    lin3d=linear_interp(x=x,y=y,z=z,f=F,dims=3)
    
    x=np.linspace(0.0,3.0,50)
    y=np.linspace(0.0,3.0,50)
    z=np.linspace(0.0,3.0,50)
    X,Y= np.meshgrid(x,y)
    F=lin3d.eval3d(x,y,z)
    ax2=fig3d.add_subplot(122)
    ax2.pcolor(X,Y,F[...,int(len(z)-1)])

    ax.set_title('Small grid')
    ax2.set_title('Large grid')
    fig3d.suptitle('3D linear interpolation of $(x+y+z)e^{-(x²+y²+z²)}$')
    fig3d.savefig('exercise2_3d_lin.pdf',dpi=200)
    plt.show()
    
#end main
    
if __name__=="__main__":
    main()
