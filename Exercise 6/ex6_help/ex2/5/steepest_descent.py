# -*- coding: utf-8 -*-
"""
Steepest descent
Created on Fri Jan 17 17:32:16 2020

@author: mvleko
"""
import numpy as np
from Gradient import gradient

'''
Define test functions
'''

def fun_3(x):
    return x[0]**2+x[1]**2+x[2]**2

def fun_3_der(x,):
    return np.array([2*x[0],2*x[1],2*x[2]])

'''
Start the steepest descent method
'''

def gamma(a,function,x,dx):
    return a/(abs(gradient(function,x,dx))+1)

def steepest_desc(function,x,dx,x_start,tolerance,a):
    g=np.ones(len(x))
    x_before=np.zeros(len(x),)
    x_v=np.zeros(len(x),)           #creates variable x
    x_before[:]=x_start[:]          #input the start value in every element
    g=gradient(function,x_start,dx)
    
    while all(g) > tolerance:   #Only if the gradient is smaller than tolerance in all dimensions, does the loop stop         
        x_v=x_before-gamma(a,fun_3,x,dx)*g
        g=gradient(function,x_v,dx)
        x_before=x_v
    return x_v
          
def variables():
    x=np.array([1.2,1,4])
    dx=0.1 
    tolerance=0.1
    x_start=np.array([2,2,1])
    a=0.1
    value=steepest_desc(fun_3,x,dx,x_start,tolerance,a)
    print('Min at ',value)
    print('Should be at [0 0 0]')
    
    
""" Run the programm """
def main():
    variables()
    
if __name__=="__main__":
    main()    
