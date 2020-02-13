# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 02:37:10 2020

Examining the electric field of a charged rod

@author: Stephen Plachta
"""

from numpy import *
from matplotlib.pyplot import *
import scipy.sparse.linalg as sla

L = 8 #metres; length of rod
Q = 5*1e-13 #Coulombs; charge on rod
Qdens = Q/L #linear charge density

cst = 4*pi*8.854*1e-12 #Farads per metre
x = linspace(-10,10,20)
y = -x
[X,Y] = meshgrid(x,y)
thR = arctan((L/2 - X)/Y) #angle from right end of rod to point in space
thL = arctan((-L/2 - X)/Y) #angle from left end of rod to point in space

#Calculate the x and y components of the electric field
Ex = Qdens*(cos(thR)-cos(thL))/(cst*abs(Y))
Ey = Qdens*(sin(thR)-sin(thL))/(cst*abs(Y))

#Plot the field near the rod
figure()
quiver(X,Y,Ex,Ey)
show()