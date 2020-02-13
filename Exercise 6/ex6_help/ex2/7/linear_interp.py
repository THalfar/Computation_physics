"""
Linear interpolation in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# l1 term from lecture slides
def l1(x): return 1-x


# l2 term from lecture slides
def l2(x): return x

# class for calculating linear interpolation for 1D, 2D and 3D cases
class linear_interp:


    def __init__(self,*args,**kwargs):
        """
        Constructor sets the 1D, 2D or 3D grid to the object.

        :param args:
        :param kwargs: x, y, z are the grid points in x, y, and z direction (numpy arrays)
                       dims is the number of dimensions
                       f is the function values at grid points (1D, 2D or 3D numpy matrices)

        """
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']  # grid in x direction
            self.f=kwargs['f']  # function values in grid points
            self.hx=np.diff(self.x)  # x intervals (does not need to be uniform)
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

    def eval1d(self,x):
        """
        Evaluates function values at points in x using linear interpolation (1D).

        :param x: numpy array of points for which interpolation is calculated
        :return: f: interpolated function values corresponding to x
        """
        if np.isscalar(x):
            x=np.array([x])
        N=len(self.x)-1
        f=np.zeros((len(x),))
        ii=0
        for val in x:
            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            else:
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*l1(t)+self.f[i+1]*l2(t)
            ii+=1
        return f

    def eval2d(self,x,y):
        """
        Evaluates function values at points in (x,y) using linear interpolation (2D).

        :param x: numpy array of points for which interpolation is calculated
        :param y: numpy array of points for which interpolation is calculated
        :return: f: function value matrix corresponding to x and y
        """
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
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                tx = (valx-self.x[i])/self.hx[i]
                ty = (valy-self.y[j])/self.hy[j]
                ptx = np.array([l1(tx),l2(tx)])
                pty = np.array([l1(ty),l2(ty)])
                A[0,:]=np.array([self.f[i,j],self.f[i,j+1]])
                A[1,:]=np.array([self.f[i+1,j],self.f[i+1,j+1]])
                f[ii,jj]=np.dot(ptx,np.dot(A,pty))
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        """
        Evaluates function values at points in (x,y,z) using linear interpolation (3D).

        :param x: numpy array of points for which interpolation is calculated
        :param y: numpy array of points for which interpolation is calculated
        :param z: numpy array of points for which interpolation is calculated
        :return: f: function value matrix at points (x,y,z)
        """
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
                    f[ii,jj,kk]=np.dot(ptx,np.dot(A,pty))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
# end class linear interp


def test_1d():
    """
    function tests eval1d interpolation with a simple x^2 function through range [1,9]
    grid interval is 0.5, interpolated interval is 0.01 (50 times more dense)
    """
    x = np.arange(1, 10, 0.5)
    y = x**2
    lin_interp_1d = linear_interp(x=x, f=y, dims=1)
    xx = np.arange(1, 9, 0.01)

    interp_values = lin_interp_1d.eval1d(xx)
    analytical_values = xx**2

    rel_err_array = 100*abs(1-interp_values/analytical_values)  # relative error array in percentage
    avg_rel_err = np.mean(rel_err_array)
    print("LINEAR 1D INTERPOLATION TEST______________________________________")
    print("Average relative error for 1d test case is: " + str(avg_rel_err) + " %")
    print("Function used: x^2")
    print("Grid used: 1:0.5:10")
    print("Interpolated to grid: 1:0.01:9\n")


def test_2d():
    """
    function tests eval1d interpolation with a simple x^2*y^2 function through range x=[1:9] and y=[1:9]
    grid interval is 0.5, interpolated interval is 0.1 (5 times more dense)
    """
    x = np.arange(0, 10, 0.5)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = X**2*Y**2
    lin_interp_2d = linear_interp(x=x,y=y, f=Z, dims=2)
    xx = np.arange(1, 9, 0.1)
    yy = xx

    interp_values = lin_interp_2d.eval2d(xx, yy)
    XX, YY = np.meshgrid(xx, yy)
    analytical_values = XX**2*YY**2

    rel_err_array = 100*abs(1-interp_values/analytical_values)  # relative error array in percentage
    avg_rel_err = np.mean(rel_err_array)
    print("LINEAR 2D INTERPOLATION TEST______________________________________")
    print("Average relative error for 2d test case is: " + str(avg_rel_err) + " %")
    print("Function used: x^2*y^2")
    print("Grid used: 1:0.5:10 (for both x and y)")
    print("Interpolated to grid: 1:0.1:9 (for both x and y)\n")


def test_3d():
    """
    function tests eval1d interpolation with a simple x^2*y^2*z^2 function through range x,y,z=[1:3]
    grid interval is 0.5, interpolated interval is 0.1 (5 times more dense)
    """
    x = np.arange(0, 3, 0.5)
    y = x
    z = x
    X, Y, Z = np.meshgrid(x, y, z)
    F = X ** 2 * Y ** 2 * Z ** 2
    linear_interp_3d = linear_interp(x=x, y=y, z=z, f=F, dims=3)
    xx = np.arange(1, 2.9, 0.1)
    yy = xx
    zz = xx

    interp_values = linear_interp_3d.eval3d(xx, yy, zz)
    XX, YY, ZZ = np.meshgrid(xx, yy, zz)
    analytical_values = XX ** 2 * YY ** 2 * ZZ ** 2

    rel_err_array = 100 * abs(1 - interp_values / analytical_values)  # relative error array in percentage
    avg_rel_err = np.mean(rel_err_array)
    print("LINEAR 3D INTERPOLATION TEST______________________________________")
    print("Average relative error for 3d test case is: " + str(avg_rel_err) + " %")
    print("Function used: x^2*y^2*z^2")
    print("Grid used: 1:0.5:3 (for x, y and z)")
    print("Interpolated to grid: 1:0.1:3 (for x, y and z)\n")


def main():
    # run tests
    test_1d()
    test_2d()
    test_3d()

    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)

    # 1d example
    x=np.linspace(0.,2.*np.pi,10)
    y=np.sin(x)
    lin1d=linear_interp(x=x,f=y,dims=1)
    xx=np.linspace(0.,2.*np.pi,100)
    ax1d.plot(xx,lin1d.eval1d(xx))
    ax1d.plot(x,y,'o',xx,np.sin(xx),'r--')
    ax1d.set_title('function')

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
    ax2d.plot_wireframe(X,Y,Z)
    ax2d3.pcolor(X,Y,Z)
    #ax2d3.contourf(X,Y,Z)

    lin2d=linear_interp(x=x,y=y,f=Z,dims=2)
    x=np.linspace(-2.0,2.0,51)
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y)
    Z = lin2d.eval2d(x,y)
     
    ax2d2.plot_wireframe(X,Y,Z)
    ax2d4.pcolor(X,Y,Z)
    
    # 3d example
    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,10)
    z=np.linspace(0.0,3.0,10)
    X,Y,Z = np.meshgrid(x,y,z)
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= np.meshgrid(x,y)
    fig3d=plt.figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)/2)])
    lin3d=linear_interp(x=x,y=y,z=z,f=F,dims=3)
    
    x=np.linspace(0.0,3.0,50)
    y=np.linspace(0.0,3.0,50)
    z=np.linspace(0.0,3.0,50)
    X,Y= np.meshgrid(x,y)
    F=lin3d.eval3d(x,y,z)
    ax2=fig3d.add_subplot(122)
    ax2.pcolor(X,Y,F[...,int(len(z)/2)])

    plt.show()
# end main


if __name__=="__main__":
    main()
