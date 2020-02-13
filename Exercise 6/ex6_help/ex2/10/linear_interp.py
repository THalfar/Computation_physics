"""
Linear interpolation in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019

20.1.2020 Minor changes by Pauli Huusari, 234214
pauli.huusari@tuni.fi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def l1(x):
   return 1-x

def l2(x):
   return x

class linear_interp:

    def __init__(self,*args,**kwargs):
        '''
        Constructor returns a grid for linear interpolation.
        Function diff from numpy calculates interval lengths.

        :param args: none
        :param kwargs: axes and function
        '''
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
      
    def eval1d(self,x):
        if np.isscalar(x): # If input is a single point.
            x=np.array([x])
        N=len(self.x)-1
        f=np.zeros((len(x),)) # This pre-allocates memory.
        ii=0
        for val in x:
            # Index i is the lower index of the grid interval.
            # It can be considered as ith data point.

            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            else:
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*l1(t)+self.f[i+1]*l2(t)
            ii+=1
        return f

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

# Test cases. The limit is set to 10**(-2).

def test_1d():
    x = np.linspace(1., 2., 20)
    y1 = x ** 2
    lin1d = linear_interp(x = x, f = y1, dims = 1)
    test_value = lin1d.eval1d(1.25)
    return abs(test_value - 25/16) < 10**(-2)

def test_2d():
    x = np.linspace(0.9, 1.2, 250)
    y = np.linspace(1.4, 1.6, 250)
    X, Y = np.meshgrid(x,y)
    Z = np.cos(X)*Y
    lin2d=linear_interp(x = x, y = y, f = Z, dims = 2)
    test_value = lin2d.eval2d(np.pi/3,1.5)[0][0]
    return abs(test_value - 3/4) < 10**(-2)

def test_3d():
    x = np.linspace(0.25, 1.3, 250)
    y = np.linspace(2.7, 3.8, 250)
    z = np.linspace(-0.1, 0.6, 250)
    X, Y, Z = np.meshgrid(x, y, z)
    F = np.exp(X)*np.cos(Y)*(Z**2)
    lin3d=linear_interp(x = x, y = y, z = z, f = F, dims = 3)
    test_value = lin3d.eval3d(np.log(2),np.pi,0.25)[0][0][0]
    return abs(test_value + 1/8) < 10**(-2)

def main():
    if test_1d():
        print('1d linear interpolation works.')
    if test_2d():
        print('2d linear interpolation works.')
    if test_3d():
        print('3d linear interpolation works.')

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
#end main
    
if __name__=="__main__":
    main()
