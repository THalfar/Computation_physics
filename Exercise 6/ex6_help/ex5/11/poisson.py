#! /usr/bin/env python3
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import time


def Jacobi_update(Phi,rho,h,N):
    Phi_new=1.0*Phi
    for i in range(1,N-1):
        for j in range(1,N-1):
            Phi_new[i,j]=0.25*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1]+h**2*rho[i,j])

    return Phi_new

def Gauss_Seidel_update(Phi,rho,h,N):
    for i in range(1,N-1):
        for j in range(1,N-1):
            Phi[i,j]=0.25*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1]+h**2*rho[i,j])

    return Phi

def Gauss_Seidel_relax_update(Phi,rho,h,N,omega):
    Phi_old=1.0*Phi
    for i in range(1,N-1):
        for j in range(1,N-1):
            Phi[i,j]=(1.0-omega)*Phi_old[i,j]+omega*0.25*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1]+h**2*rho[i,j])

    return Phi
    
def desired_accuracy(Phi_old,Phi_new,tolerance):
    if (amax(abs(Phi_old-Phi_new))/abs(amax(Phi_new))<tolerance):
        return True
    else:
        return False
            
L=1.0
h=0.05
N=int(L/h+1)

x=y=linspace(0.,L,N)
rho=zeros((N,N))
Phi=zeros((N,N))
Phi[:,-1]=1.0
X,Y=meshgrid(x,y)
#rho=exp(-((X-L/2)**2+(Y-L/2)**2)/0.001)

#print(len(x))

fig = figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(transpose(X),transpose(Y),Phi,rstride=1,cstride=1)
ax.set_xlabel('x')
ax.set_ylabel('y')


good_accuracy=False
tolerance=1.0e-4
i=1
omega=1.7
while not good_accuracy:
    Phi_old=1.0*Phi
    Phi=Jacobi_update(Phi,rho,h,N)
    #Phi=Gauss_Seidel_update(Phi,rho,h,N)
    #Phi=Gauss_Seidel_relax_update(Phi,rho,h,N,omega)
    good_accuracy=desired_accuracy(Phi_old,Phi,tolerance)
    i+=1
    
    #print(i)
print('Total number of loops',i)
fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(transpose(X),transpose(Y),Phi,rstride=1,cstride=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
show()
    
