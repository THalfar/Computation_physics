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

TESTPOINTS = int(1e5)
TESTPOINTS2D = 500
TESTPOINTS3D = 75

class linear_interp:

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
    
    def l1(self, t):
        return 1-t
    
    def l2(self, t):
        return t
        
    def eval1d(self,x):
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
                f[ii]=self.f[i]*self.l1(t)+self.f[i+1]*self.l2(t)
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
                ptx = np.array([self.l1(tx),self.l2(tx)])
                pty = np.array([self.l1(ty),self.l2(ty)])
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
                    ptx = np.array([self.l1(tx),self.l2(tx)])
                    pty = np.array([self.l1(ty),self.l2(ty)])
                    ptz = np.array([self.l1(tz),self.l2(tz)])
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

# Plots an error curve for 1D linear interpolation case
def test1D():
    
    # Testing function for 1D case
    def fun(x): return x**3 - 2*x
        
    
    test_x = np.linspace(-2,2,TESTPOINTS)
    true_y = fun(test_x)
    
    errors = []
    points = np.geomspace(10,1e3, 10, dtype = int)
    for point in points:
        
        x = np.linspace(-2,2, point)
        y = fun(x)
        lin1d = linear_interp(x=x, f=y, dims=1)
        
        test_lin = lin1d.eval1d(test_x)
        
        errors.append(np.mean(np.abs(test_lin - true_y)))
    # Plotting error
    fig = plt.figure()      
    plt.plot(points,errors, 'or', label = r'f(x) = x^3-2x')               
    plt.xlabel("Interpolation points")
    plt.ylabel("Mean abs. error")        
    plt.title("Linear 1D interpolation error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    fig.savefig('lin_inter_1D.pdf', dpi = 200)

# Plots an error curve for 2D linear interpolation case    
def test2D():
    
    def fun(x,y): return y*2**(x+y)
    
    x_test = np.linspace(-2,2,TESTPOINTS2D)
    y_test = np.linspace(-2,2,TESTPOINTS2D)
    X,Y = np.meshgrid(x_test, y_test)
    true_Z = fun(X,Y)
        
    errors = []
    points = np.geomspace(3,100,10, dtype=int)
    for point in points:
        x = np.linspace(-2,2,point)
        y = np.linspace(-2,2,point)
        X,Y = np.meshgrid(x,y)
        Z = fun(X,Y)
    
        lin2d = linear_interp(x=x, y=y, f=Z, dims = 2)
        test_Z = lin2d.eval2d(x_test, y_test)
        errors.append(np.mean(np.abs(test_Z - true_Z)))
    #Plotting error
    fig = plt.figure()      
    plt.plot(points,errors, 'or', label = r'f(x,y) = y*2^(x+y')               
    plt.xlabel("Interpolation points")
    plt.ylabel("Mean abs. error")        
    plt.title("Linear 2D interpolation error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    fig.savefig('lin_inter_2D.pdf', dpi = 200)

# Plots an error curve for 3D linear interpolation case    
def test3D():

    def fun(x,y,z): return np.exp(np.sqrt(x**2+y**2+z**2))
    
    x_test = np.linspace(-2,2,TESTPOINTS3D)
    y_test = np.linspace(-2,2,TESTPOINTS3D)
    z_test = np.linspace(-2,2,TESTPOINTS3D)
    X,Y,Z = np.meshgrid(x_test, y_test, z_test)
    true_F = fun(X,Y,Z)
    
    errors = []
    points = np.geomspace(3,30,10,dtype = int)
    for point in points:
        x = np.linspace(-2,2,point)
        y = np.linspace(-2,2,point)
        z = np.linspace(-2,2,point)
        X,Y,Z = np.meshgrid(x,y,z)
        F = fun(X,Y,Z)
        
        lin3d = linear_interp(x=x,y=y,z=z,f=F,dims=3)
        test_F = lin3d.eval3d(x_test,y_test,z_test)
        errors.append(np.mean(np.abs(true_F - test_F)))
       #Plotting error
    fig = plt.figure()      
    plt.plot(points,errors, 'or', label = r'f(x,y,z) = exp(x^2+y^2+z^2)')               
    plt.xlabel("Interpolation points")
    plt.ylabel("Mean abs. error")        
    plt.title("Linear 3D interpolation error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    fig.savefig('lin_inter_3D.pdf', dpi = 200)
    
   
    
def main():
    test1D()
    test2D()
    test3D()
    
if __name__=="__main__":
    main()
