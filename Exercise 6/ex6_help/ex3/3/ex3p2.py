"""
2d interpolation revisited

FYS-4096 Computational Physics
exercise 3 assignment, problem 2

By Stephen Plachta, 24 January 2019
"""

from numpy import *
#import numpy as *
from matplotlib.pyplot import *
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spline_class import *

xmin = -2.0
ymin = -2.0
xmax = 2.0
ymax = 2.0
N = 30
x = linspace(xmin,xmax,N)
y = linspace(ymin,ymax,N)
[X,Y] = meshgrid(x,y)

def fun(x,y):
    f = (x + y)*exp(-sqrt(x*x + y*y))
    return f

Z = fun(X,Y) #get "experimental data"
Zsp = spline(x=x,y=y,f=Z,dims=2) #interpolate

N = 100
xmax = ymax/sqrt(1.75)
xr = linspace(0,xmax,N)
yr = sqrt(1.75)*xr #straight line

r = Zsp.eval2d(xr,yr) 
rr = diag(r) #estimated values from data along straight line

figure()
scatter(xr,rr,8,color='red')
Zex = fun(xr,yr) #exact values
plot(xr,Zex)
show()
#print(r)
