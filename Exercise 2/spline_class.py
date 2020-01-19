"""
Cubic hermite splines in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

from numpy import *
import matplotlib.pyplot as plt

TESTPOINTS = int(1e5)
TESTPOINTS2D = 500
TESTPOINTS3D = 75


def init_1d_spline(x,f,h):
    # now using complete boundary conditions
    # with forward/backward derivative
    # - natural boundary conditions commented
    a=zeros((len(x),))
    b=zeros((len(x),))
    c=zeros((len(x),))
    d=zeros((len(x),))
    fx=zeros((len(x),))

    # a[0]=1.0 # not needed
    b[0]=1.0

    # natural boundary conditions 
    #c[0]=0.5
    #d[0]=1.5*(f[1]-f[0])/(x[1]-x[0])

    # complete boundary conditions
    c[0]=0.0
    d[0]=(f[1]-f[0])/(x[1]-x[0])
    
    for i in range(1,len(x)-1):
        d[i]=6.0*(h[i]/h[i-1]-h[i-1]/h[i])*f[i]-6.0*h[i]/h[i-1]*f[i-1]+6.0*h[i-1]/h[i]*f[i+1]
        a[i]=2.0*h[i]
        b[i]=4.0*(h[i]+h[i-1])
        c[i]=2.0*h[i-1]        
    #end for

    
    b[-1]=1.0
    #c[-1]=1.0 # not needed

    # natural boundary conditions
    #a[-1]=0.5
    #d[-1]=1.5*(f[-1]-f[-2])/(x[-1]-x[-2])

    # complete boundary conditions
    a[-1]=0.0
    d[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    
    # solve tridiagonal eq. A*f=d
    c[0]=c[0]/b[0]
    d[0]=d[0]/b[0]
    for i in range(1,len(x)-1):
        temp=b[i]-c[i-1]*a[i]
        c[i]=c[i]/temp
        d[i]=(d[i]-d[i-1]*a[i])/temp
    #end for
        
    fx[-1]=d[-1]
    for i in range(len(x)-2,-1,-1):
        fx[i]=d[i]-c[i]*fx[i+1]
    #end for
        
    return fx
# end function init_1d_spline

""" 
Add smoothing functions 

def smooth1d(x,f,factor=3):
    ...
    ...
    return ...

def smooth2d(x,y,f,factor=3):
    ...
    ...
    return ... 

def smooth3d(x,y,z,f,factor=3):
    ...
    ...
    ...
    return ...
"""

class spline:

    def __init__(self,*args,**kwargs):
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.fx=init_1d_spline(self.x,self.f,self.hx)
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.hy=diff(self.y)
            self.fx=zeros(shape(self.f))
            self.fy=zeros(shape(self.f))
            self.fxy=zeros(shape(self.f))
            for i in range(max([len(self.x),len(self.y)])):
                if (i<len(self.y)):
                    self.fx[:,i]=init_1d_spline(self.x,self.f[:,i],self.hx)
                if (i<len(self.x)):
                    self.fy[i,:]=init_1d_spline(self.y,self.f[i,:],self.hy)
            #end for
            for i in range(len(self.y)):
                self.fxy[:,i]=init_1d_spline(self.x,self.fy[:,i],self.hx)
            #end for
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.hy=diff(self.y)
            self.hz=diff(self.z)
            self.fx=zeros(shape(self.f))
            self.fy=zeros(shape(self.f))
            self.fz=zeros(shape(self.f))
            self.fxy=zeros(shape(self.f))
            self.fxz=zeros(shape(self.f))
            self.fyz=zeros(shape(self.f))
            self.fxyz=zeros(shape(self.f))
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fx[:,i,j]=init_1d_spline(self.x,self.f[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fy[i,:,j]=init_1d_spline(self.y,self.f[i,:,j],self.hy)
                    if (i<len(self.x) and j<len(self.y)):
                        self.fz[i,j,:]=init_1d_spline(self.z,self.f[i,j,:],self.hz)
            #end for
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxy[:,i,j]=init_1d_spline(self.x,self.fy[:,i,j],self.hx)
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxz[:,i,j]=init_1d_spline(self.x,self.fz[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fyz[i,:,j]=init_1d_spline(self.y,self.fz[i,:,j],self.hy)
            #end for
            for i in range(len(self.y)):
                for j in range(len(self.z)):
                    self.fxyz[:,i,j]=init_1d_spline(self.x,self.fyz[:,i,j],self.hx)
            #end for
        else:
            print('Either dims is missing or specific dims is not available')
        #end if
    
    def p1(self,t): return (1+2*t)*(t-1)**2
    def p2(self,t): return t**2 * (3-2*t)
    def q1(self,t): return t*(t-1)**2
    def q2(self,t): return t**2 * (t-1)
        
            
    def eval1d(self,x):
        if isscalar(x):
            x=array([x])
        N=len(self.x)-1
        f=zeros((len(x),))
        ii=0
        for val in x:
            i=floor(where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            else:
                t=(val-self.x[i])/self.hx[i]                
                f[ii]=self.f[i]*self.p1(t)+self.f[i+1]*self.p2(t)+self.hx[i]*(self.fx[i]*self.q1(t)+self.fx[i+1]*self.q2(t))
            ii+=1

        return f
    #end eval1d

    def eval2d(self,x,y):
        if isscalar(x):
            x=array([x])
        if isscalar(y):
            y=array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        f=zeros((len(x),len(y)))
        A=zeros((4,4))
        ii=0
        for valx in x:
            i=floor(where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=floor(where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                u = (valx-self.x[i])/self.hx[i]
                v = (valy-self.y[j])/self.hy[j]
                pu = array([self.p1(u),self.p2(u),self.hx[i]*self.q1(u),self.hx[i]*self.q2(u)])
                pv = array([self.p1(v),self.p2(v),self.hy[j]*self.q1(v),self.hy[j]*self.q2(v)])
                A[0,:]=array([self.f[i,j],self.f[i,j+1],self.fy[i,j],self.fy[i,j+1]])
                A[1,:]=array([self.f[i+1,j],self.f[i+1,j+1],self.fy[i+1,j],self.fy[i+1,j+1]])
                A[2,:]=array([self.fx[i,j],self.fx[i,j+1],self.fxy[i,j],self.fxy[i,j+1]])
                A[3,:]=array([self.fx[i+1,j],self.fx[i+1,j+1],self.fxy[i+1,j],self.fxy[i+1,j+1]])           
                
                f[ii,jj]=dot(pu,dot(A,pv))
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        if isscalar(x):
            x=array([x])
        if isscalar(y):
            y=array([y])
        if isscalar(z):
            z=array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=zeros((len(x),len(y),len(z)))
        A=zeros((4,4))
        B=zeros((4,4))
        ii=0
        for valx in x:
            i=floor(where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=floor(where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                for valz in z:
                    k=floor(where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1
                    u = (valx-self.x[i])/self.hx[i]
                    v = (valy-self.y[j])/self.hy[j]
                    t = (valz-self.z[k])/self.hz[k]
                    pu = array([self.p1(u),self.p2(u),self.hx[i]*self.q1(u),self.hx[i]*self.q2(u)])
                    pv = array([self.p1(v),self.p2(v),self.hy[j]*self.q1(v),self.hy[j]*self.q2(v)])
                    pt = array([self.p1(t),self.p2(t),self.hz[k]*self.q1(t),self.hz[k]*self.q2(t)])
                    B[0,:]=array([self.f[i,j,k],self.f[i,j,k+1],self.fz[i,j,k],self.fz[i,j,k+1]])
                    B[1,:]=array([self.f[i+1,j,k],self.f[i+1,j,k+1],self.fz[i+1,j,k],self.fz[i+1,j,k+1]])
                    B[2,:]=array([self.fx[i,j,k],self.fx[i,j,k+1],self.fxz[i,j,k],self.fxz[i,j,k+1]])
                    B[3,:]=array([self.fx[i+1,j,k],self.fx[i+1,j,k+1],self.fxz[i+1,j,k],self.fxz[i+1,j,k+1]])
                    A[:,0]=dot(B,pt)
                    B[0,:]=array([self.f[i,j+1,k],self.f[i,j+1,k+1],self.fz[i,j+1,k],self.fz[i,j+1,k+1]])
                    B[1,:]=array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1],self.fz[i+1,j+1,k],self.fz[i+1,j+1,k+1]])
                    B[2,:]=array([self.fx[i,j+1,k],self.fx[i,j+1,k+1],self.fxz[i,j+1,k],self.fxz[i,j+1,k+1]])
                    B[3,:]=array([self.fx[i+1,j+1,k],self.fx[i+1,j+1,k+1],self.fxz[i+1,j+1,k],self.fxz[i+1,j+1,k+1]])
                    A[:,1]=dot(B,pt)

                    B[0,:]=array([self.fy[i,j,k],self.fy[i,j,k+1],self.fyz[i,j,k],self.fyz[i,j,k+1]])
                    B[1,:]=array([self.fy[i+1,j,k],self.fy[i+1,j,k+1],self.fyz[i+1,j,k],self.fyz[i+1,j,k+1]])
                    B[2,:]=array([self.fxy[i,j,k],self.fxy[i,j,k+1],self.fxyz[i,j,k],self.fxyz[i,j,k+1]])
                    B[3,:]=array([self.fxy[i+1,j,k],self.fxy[i+1,j,k+1],self.fxyz[i+1,j,k],self.fxyz[i+1,j,k+1]])
                    A[:,2]=dot(B,pt)
                    B[0,:]=array([self.fy[i,j+1,k],self.fy[i,j+1,k+1],self.fyz[i,j+1,k],self.fyz[i,j+1,k+1]])
                    B[1,:]=array([self.fy[i+1,j+1,k],self.fy[i+1,j+1,k+1],self.fyz[i+1,j+1,k],self.fyz[i+1,j+1,k+1]])
                    B[2,:]=array([self.fxy[i,j+1,k],self.fxy[i,j+1,k+1],self.fxyz[i,j+1,k],self.fxyz[i,j+1,k+1]])
                    B[3,:]=array([self.fxy[i+1,j+1,k],self.fxy[i+1,j+1,k+1],self.fxyz[i+1,j+1,k],self.fxyz[i+1,j+1,k+1]])
                    A[:,3]=dot(B,pt)
                
                    f[ii,jj,kk]=dot(pu,dot(A,pv))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
#end class spline

# Plots an error curve for 1D spline interpolation case
def test1D():
    
    # Testing function for 1D case
    def fun(x): return x**3 - 2*x
            
    test_x = linspace(-2,2,TESTPOINTS)
    true_y = fun(test_x)
    
    errors = []
    points = geomspace(10,1e3, 10, dtype = int)
    for point in points:
        
        x = linspace(-2,2, point)
        y = fun(x)
        splin1d = spline(x=x, f=y, dims=1)        
        test_lin = splin1d.eval1d(test_x)        
        errors.append(mean(abs(test_lin - true_y)))
        
    # Plotting error
    fig = plt.figure()      
    plt.plot(points,errors, 'or', label = r'f(x) = x^3-2x')               
    plt.xlabel("Interpolation points")
    plt.ylabel("Mean abs. error")        
    plt.title("Spline 1D interpolation error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    fig.savefig('spline_inter_1D.pdf', dpi = 200)
    
# Plots an error curve for 2D spline interpolation case    
def test2D():
    
    def fun(x,y): return y*2**(x+y)
    
    x_test = linspace(-2,2,TESTPOINTS2D)
    y_test = linspace(-2,2,TESTPOINTS2D)
    X,Y = meshgrid(x_test, y_test)
    true_Z = fun(X,Y)
        
    errors = []
    points = geomspace(3,100,10, dtype=int)
    for point in points:
        x = linspace(-2,2,point)
        y = linspace(-2,2,point)
        X,Y = meshgrid(x,y)
        Z = fun(X,Y)
    
        splin2d = spline(x=x, y=y, f=Z, dims = 2)
        test_Z = splin2d.eval2d(x_test, y_test)
        errors.append(mean(abs(test_Z - true_Z)))
    #Plotting error
    fig = plt.figure()      
    plt.plot(points,errors, 'or', label = r'f(x,y) = y*2^(x+y')               
    plt.xlabel("Interpolation points")
    plt.ylabel("Mean abs. error")        
    plt.title("Spline 2D interpolation error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    fig.savefig('spline_inter_2D.pdf', dpi = 200)    
    
# Plots an error curve for 3D spline interpolation case    
def test3D():

    def fun(x,y,z): return exp(sqrt(x**2+y**2+z**2))
    
    x_test = linspace(-2,2,TESTPOINTS3D)
    y_test = linspace(-2,2,TESTPOINTS3D)
    z_test = linspace(-2,2,TESTPOINTS3D)
    X,Y,Z = meshgrid(x_test, y_test, z_test)
    true_F = fun(X,Y,Z)
    
    errors = []
    points = geomspace(3,30,10,dtype = int)
    for point in points:
        x = linspace(-2,2,point)
        y = linspace(-2,2,point)
        z = linspace(-2,2,point)
        X,Y,Z = meshgrid(x,y,z)
        F = fun(X,Y,Z)
        
        splin3d = spline(x=x,y=y,z=z,f=F,dims=3)
        test_F = splin3d.eval3d(x_test,y_test,z_test)
        errors.append(mean(abs(true_F - test_F)))
       #Plotting error
    fig = plt.figure()      
    plt.plot(points,errors, 'or', label = r'f(x,y,z) = exp(x^2+y^2+z^2)')               
    plt.xlabel("Interpolation points")
    plt.ylabel("Mean abs. error")        
    plt.title("Spline 3D interpolation error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    fig.savefig('spline_inter_3D.pdf', dpi = 200)
    
          
def main():
    
    test1D()
    test2D()
    test3D()

if __name__=="__main__":
    main()
