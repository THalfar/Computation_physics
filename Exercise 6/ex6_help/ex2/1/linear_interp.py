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


"""
Add basis functions l1 and l2 here
"""
def l1(t):
    """
    Function l1 for linear interpolation based on formula 1 on lecture 2 slides
    :param t:  function parameter
    :return: function value
    """
    return 1-t

def l2(t):
    """
     Function l2 for linear interpolation based on formula 1 on lecture 2 slides
     :param t:  function parameter
     :return: function value
     """
    return t

class linear_interp:
    """
    class for linear interpolation in 1d, 2d, and 3d
    """
    def __init__(self,*args,**kwargs):
        """
        defining arguments for 1d, 2d and 3d interpolation
        :param args:
        :param kwargs: coordinates and function
        """
        self.dims=kwargs['dims']
        # 1d
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
        # 2d
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
        # 3d
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
            self.hz=np.diff(self.z)
        # something wrong with dimensions
        else:
            print('Either dims is missing or specific dims is not available')
      
    def eval1d(self,x):
        """
        calculating 1d linear interpolation
        :param x: x values over interval
        :return: values of function at given x coordinates
        """
        # checks if x values are scalars
        if np.isscalar(x):
            x=np.array([x])
        # number of gaps between points
        N=len(self.x)-1
        # empty array for function values
        f=np.zeros((len(x),))
        ii=0
        # looping through all x coordinates
        for val in x:
            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            # linear interpolation based on formula 1 on lecture 2 slides
            else:
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*l1(t)+self.f[i+1]*l2(t)
            ii+=1
        return f

    def eval2d(self,x,y):
        """
         calculating 2d linear interpolation
         :param x: x and y values over interval
         :return: values of function at given x,y coordinates
         """
        # checks if x and y values are scalars
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        # number of gaps between points for x and y coordinates
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        # empty array for function values and empty 2x2 matrix for formula 4 on lecture 2 slides
        f=np.zeros((len(x),len(y)))
        A=np.zeros((2,2))
        ii=0
        # looping through all x coordinates
        for valx in x:
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            # looping through all y coordinates
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
         calculating 3d linear interpolation
         :param x: x,y and z values over interval
         :return: values of function at given x,y,z coordinates
         """
        # checks if x, y and z values are scalars
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        if np.isscalar(z):
            z=np.array([z])
        # number of gaps between points for x and y coordinates
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        # empty array for function values and empty 2x2 matrix for formula 4 on lecture 2 slides and also matrix for
        # next index of matrix given at formula 4.
        f=np.zeros((len(x),len(y),len(z)))
        A=np.zeros((2,2))
        B=np.zeros((2,2))
        ii=0
        # looping through all x coordinates
        for valx in x:
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            # looping through all y coordinates
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                # looping through all z coordinates
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
    # linear interpolation with 10 points
    x = np.linspace(0., 2. * np.pi, 10)
    y = np.sin(x)
    lin1d = linear_interp(x=x, f=y, dims=1)
    # 20 points to get values for mid-points of interpolation
    xx = np.linspace(0., 2. * np.pi, 20)
    linear_values = lin1d.eval1d(xx)
    y = np.sin(xx)
    sq_diff = np.zeros(len(x))
    for i in range(len(x)):
        sq_diff[i] = (y[1+2*i]-linear_values[1+2*i])**2
    diff = np.mean(sq_diff)
    print("1D mean square error for mid-points: ", diff)

def test_2d():
    # x,y coordinates with 11 points
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 11)
    # function values with meshgrid
    X, Y = np.meshgrid(x, y)
    Z = X * np.exp(-1.0 * (X * X + Y * Y))
    # linear interpolation
    lin2d = linear_interp(x=x, y=y, f=Z, dims=2)
    # x,y coordinates with 10 points for mid-points
    xx = np.linspace(-1.8, 1.8, 10)
    yy = np.linspace(-1.8, 1.8, 10)
    X, Y = np.meshgrid(xx,yy)
    # linear interpolation values
    linear_values = lin2d.eval2d(xx, yy)
    # function values
    y = X*np.exp(-1.0*(X*X+Y*Y))
    sq_diff = []
    for i in range(len(xx)):
        for j in range(len(yy)):
            sq_diff.append((y[i,j]-linear_values[i,j])**2)
    diff = np.mean(sq_diff)
    print("2D mean square error for mid-points: ", diff)

def test_3d():
    # x,y and z coordinates with 10 points forming a grid
    x = np.linspace(0.0, 3.0, 10)
    y = np.linspace(0.0, 3.0, 10)
    z = np.linspace(0.0, 3.0, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    # function values
    F = (X + Y + Z) * np.exp(-1.0 * (X * X + Y * Y + Z * Z))
    lin3d = linear_interp(x=x, y=y, z=z, f=F, dims=3)
    # x,y and z coordinates somewhere close to mid-point of last coordinates
    xx = np.linspace(0.17, 2.83, 10)
    yy = np.linspace(0.17, 2.83, 10)
    zz = np.linspace(0.17, 2.83, 10)
    X, Y = np.meshgrid(xx, yy)
    # linear interpolation values
    linear_values = lin3d.eval3d(xx, yy, zz)
    # function values
    y = (X + Y + Z) * np.exp(-1.0 * (X * X + Y * Y + Z * Z))
    sq_diff = []
    for i in range(len(xx)):
        for j in range(len(yy)):
            for k in range(len(zz)):
                sq_diff.append((y[i,j,k]-linear_values[i,j,k])**2)
    diff = np.mean(sq_diff)
    print("3D mean square error for mid-points: ", diff)

def main():
    # creating new figure
    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)

    # 1d example
    # linear interpolation with 10 points
    x=np.linspace(0.,2.*np.pi,10)
    y=np.sin(x)
    lin1d=linear_interp(x=x,f=y,dims=1)
    # 100 points for drawing figures
    xx=np.linspace(0.,2.*np.pi,100)
    # plotting function and linear interpolation of function
    ax1d.plot(xx,lin1d.eval1d(xx))
    ax1d.plot(x,y,'o',xx,np.sin(xx),'r--')
    ax1d.set_title('function')

    # 2d example
    # creating new figure with 4 subplots
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(221, projection='3d')
    ax2d2 = fig2d.add_subplot(222, projection='3d')
    ax2d3 = fig2d.add_subplot(223)
    ax2d4 = fig2d.add_subplot(224)

    # x,y coordinates with 11 points and grid made of them
    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y)
    # calculating function values and plotting them
    Z = X*np.exp(-1.0*(X*X+Y*Y))
    ax2d.plot_wireframe(X,Y,Z)
    ax2d3.pcolor(X,Y,Z)
    #ax2d3.contourf(X,Y,Z)

    # linear interpolation from 11 points into 51 points
    lin2d=linear_interp(x=x,y=y,f=Z,dims=2)
    x=np.linspace(-2.0,2.0,51)
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y)
    Z = lin2d.eval2d(x,y)
     
    ax2d2.plot_wireframe(X,Y,Z)
    ax2d4.pcolor(X,Y,Z)
    
    # 3d example
    # x,y and z coordinates with 10 points forming a grid
    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,10)
    z=np.linspace(0.0,3.0,10)
    X,Y,Z = np.meshgrid(x,y,z)
    # plotting figure
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= np.meshgrid(x,y)
    fig3d=plt.figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)/2)])

    # calculating interpolation and plotting it
    lin3d=linear_interp(x=x,y=y,z=z,f=F,dims=3)
    x=np.linspace(0.0,3.0,50)
    y=np.linspace(0.0,3.0,50)
    z=np.linspace(0.0,3.0,50)
    X,Y= np.meshgrid(x,y)
    F=lin3d.eval3d(x,y,z)
    ax2=fig3d.add_subplot(122)
    ax2.pcolor(X,Y,F[...,int(len(z)/2)])

    plt.show()

    test_1d()
    test_2d()
    test_3d()
#end main
    
if __name__=="__main__":
    main()
