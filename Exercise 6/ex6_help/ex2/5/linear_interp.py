"""
Linear interpolation in 1d, 2d, and 3d

Related to FYS-4096 Computational Physics
exercise 2 assignment 1.

By Ilkka Kylanpaa on January 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
Basis functions l1 and l2 as defined in the lecture script week 2 p. 4
"""

def l1(t):
    return 1-t    
    
def l2(t):   
    return t

'''
Linear interpolation in 1D, 2D and 3D   
This class interpolates between input points with a linear interpolation algorithm (as introduced in the lecture slides)
'''

class linear_interp:

    def __init__(self,*args,**kwargs):
        #Initialize all needed variables
        self.dims=kwargs['dims']        #Dkwawrgs: variable length argument list
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)     #h values=spacing, also non-uniform
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
      
    def eval1d(self,x):          #Interpolation in 1D. It interpolates points linearly between the given grid
        if np.isscalar(x):
            x=np.array([x])      #Turn x into an array, if its a scalar
        N=len(self.x)-1          #Upper limit for sum
        f=np.zeros((len(x),))
        counter=0
        for val in x:            #Loop through all x values
            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)    #If ... true: [0]; false: [-1]
            if i==N:                                            #If i is out of our range (it equals the upper limit), f equals our function value for the greates value (which is N)
                f[counter]=self.f[i]
            else:                                               #If not, we interpolate linearly between the two consecutive points i and i+1
                t=(val-self.x[i])/self.hx[i]                    #Create a new point at 1/h distance to the two consecutive points
                f[counter]=self.f[i]*l1(t)+self.f[i+1]*l2(t)    #Add the point to our values
            counter+=1          #iteration counter
        return f

    def eval2d(self,x,y):       #Interpolation in 2D
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        Nx=len(self.x)-1        #Upper limit for x-interpolation
        Ny=len(self.y)-1        #Upper limit for y-interpolation
        f=np.zeros((len(x),len(y))) #Initialize matrices
        A=np.zeros((2,2))
        ii=0                    #counter 1
        for valx in x:          #Loop through all x values
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):         #If we reach the upper limit of x: set our counter 1 to -1 
                i-=1
            jj=0                #counter 2
            for valy in y:      #Loop through all y values
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                tx = (valx-self.x[i])/self.hx[i]        #Calculate the position of a new point at eg half the distance for h=2 in the rectangle created in the two dimensions by 2x and 2y points
                ty = (valy-self.y[j])/self.hy[j]
                ptx = np.array([l1(tx),l2(tx)])
                pty = np.array([l1(ty),l2(ty)])
                A[0,:]=np.array([self.f[i,j],self.f[i,j+1]])        #Extract row 0
                A[1,:]=np.array([self.f[i+1,j],self.f[i+1,j+1]])    #Row 1
                f[ii,jj]=np.dot(ptx,np.dot(A,pty))                  #Add the point to our values
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):                     #3d interpolation, does the same as before, but in 3d
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        if np.isscalar(z):
            z=np.array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=np.zeros((len(x),len(y),len(z)))      #Initialize matrices
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
    
'''
Tests + test functions
The tests compare the calculated value with analytical values for a specific function
'''
def fun1d(x):
    return np.sin(x)

def fun2d(x,y):
    return np.sin(x)+np.cos(y)

def fun3d(x,y,z):
    return np.sin(x)+np.cos(y)-np.sin(z)

def test_1d_interpolation():
    x=np.linspace(0.,2.*np.pi,10)
    f=fun1d(x)
    
    lin=linear_interp(x=x,f=f,dims=1)     #Create the interpolation class
    y_num=lin.eval1d(x)
    y_ana=np.sin(np.linspace(0.,2.*np.pi,len(y_num)))
    if all(abs(y_ana-y_num))<0.001:
        print('1d working')
    else:
        print('1d not working')
 
def test_2d_interpolation():
    x=np.linspace(0.,2.*np.pi,10)
    y=np.linspace(0.,2.*np.pi,10)
    X,Y = np.meshgrid(x, y)                 #Creates a 2d grid 
    z=fun2d(X,Y)
    
    lin=linear_interp(x=x,y=y,f=z,dims=2)     #Create the interpolation class
    f_num=lin.eval2d(x,y)
    xa=np.linspace(0.,2.*np.pi,np.size(f_num,0))
    ya=np.linspace(0.,2.*np.pi,np.size(f_num,1))
    Xa,Ya=np.meshgrid(xa,ya)
    f_ana=fun2d(Xa,Ya)
    if np.all(abs(f_ana-f_num))<0.001:
        print('2d working')
    else:
        print('2d not working')  

def test_3d_interpolation():
    x=np.linspace(0.,2.*np.pi,10)
    y=np.linspace(0.,2.*np.pi,10)
    z=np.linspace(0.,2.*np.pi,10)
    X,Y,Z = np.meshgrid(x, y,z)
    f=fun3d(X,Y,Z)
    
    lin=linear_interp(x=x,y=y,z=z,f=f,dims=3)     #Create the interpolation class
    f_num=lin.eval3d(x,y,z)
    xa=np.linspace(0.,2.*np.pi,np.size(f_num,0))
    ya=np.linspace(0.,2.*np.pi,np.size(f_num,1))
    za=np.linspace(0.,2.*np.pi,np.size(f_num,2))
    Xa,Ya, Za=np.meshgrid(xa,ya,za)
    f_ana=fun3d(Xa,Ya,Za)
    if np.all(abs(f_ana-f_num))<0.001:
        print('3d working')
    else:
        print('3d not working')  
     
    
'''
Plots
'''    

    
def main():
    test_1d_interpolation()
    test_2d_interpolation()
    test_3d_interpolation()

    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)

    # 1d example
    x=np.linspace(0.,2.*np.pi,10)
    y=np.sin(x)
    lin1d=linear_interp(x=x,f=y,dims=1)
    xx=np.linspace(0.,2.*np.pi,100)
    ax1d.plot(xx,lin1d.eval1d(xx),'g')
    ax1d.plot(x,y,'ob', markersize=15, markerfacecolor='None')
    ax1d.plot(xx,np.sin(xx),'r--')
#    ax1d.set_title('function')
    ax1d.set(xlabel='x', ylabel='y', title='1d interpolation')
    ax1d.grid()
#    plt.savefig('1Dlinear_inter.png',dpi=200)

    # 2d example
    fig2d = plt.figure()
    plt.tight_layout() #These values surpress the cropping of the axes
#    plt.tight_layout(pad=1, w_pad=0.001, h_pad=0.01) #These values surpress the cropping of the axes
    ax2d = fig2d.add_subplot(221, projection='3d')
    ax2d2 = fig2d.add_subplot(222, projection='3d')
#    ax2d3 = fig2d.add_subplot(223)
#    ax2d4 = fig2d.add_subplot(224)

    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y)
    Z = X*np.exp(-1.0*(X*X+Y*Y))
    ax2d.plot_wireframe(X,Y,Z)
    ax2d.set(xlabel='x', ylabel='y', zlabel='z', title='without interpolation')
    ax2d2.set(xlabel='x', ylabel='y', zlabel='z', title='with interpolation')
#    ax2d3.pcolor(X,Y,Z)
#    ax2d3.contourf(X,Y,Z)

    lin2d=linear_interp(x=x,y=y,f=Z,dims=2)
    x=np.linspace(-2.0,2.0,51)
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y)
    Z = lin2d.eval2d(x,y)
     
    ax2d2.plot_wireframe(X,Y,Z)    
#    ax2d4.pcolor(X,Y,Z)
#    plt.savefig('2Dlinear_inter.png',dpi=200)
    
    
    # 3d example
    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,10)
    z=np.linspace(0.0,3.0,10)
    X,Y,Z = np.meshgrid(x,y,z)
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= np.meshgrid(x,y)
    fig3d=plt.figure()
    plt.tight_layout() #These values surpress the cropping of the axes
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
    ax.set(xlabel='x', ylabel='y', title='without interpolation')
    ax2.set(xlabel='x', ylabel='y', title='with interpolation')
    ax2.yaxis.tick_right()                  #set the y-axis to the right to prevent the overlapp with subplot 1
    ax2.yaxis.set_label_position("right")
#    plt.savefig('3Dlinear_inter.png',dpi=200)

    plt.show()
#end main
    
if __name__=="__main__":
    main()
