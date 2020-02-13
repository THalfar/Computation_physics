"""
2d integral revisited

FYS-4096 Computational Physics
exercise 3 assignment, problem 1

By Stephen Plachta, 24 January 2019
"""

from numpy import *
#import numpy as *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps

valWA = 1.57348 #value from WolframAlpha double integral calculator

def fun(x,y):
    """function to analyze"""
    f = (x + y)*exp(-sqrt(x*x + y*y))
    return f

def simp_appx(x,y,X,Y):
    """estimate value of integral of funtion defined in fun(x,y)"""
    val = simps(simps(fun(X,Y),dx=x[1]-x[0],axis=0), dx=y[1]-y[0])
    return val

x = linspace(0,2,100)
y = linspace(-2,2,100)
[X,Y] = meshgrid(x,y)

val = simp_appx(x,y,X,Y)
print("100x100 grid")
print("estimate = ")
print(val)
print("error = ") 
print(abs(val-valWA))

x2 = linspace(0,2,10)
y2 = linspace(-2,2,10)
[X2,Y2] = meshgrid(x2,y2)
val2 = simp_appx(x2,y2,X2,Y2)
print("")
print("10x10 grid")
print("estimate = ")
print(val2)
print("error = ") 
print(abs(val2-valWA))
