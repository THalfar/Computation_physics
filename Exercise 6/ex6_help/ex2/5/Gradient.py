# -*- coding: utf-8 -*-
"""
This script calculates the n-dim gradient of a n-dim function
It outputs whether it is working or not by comparing the result with an analytical test function
Exercise sheet 2 number 3
Created on Fri Jan 17 16:21:57 2020

@author: mvleko
"""

import numpy as np

'''
Define test functions
'''

def fun_3(x):
    return x[0]**2+x[1]**2+x[2]**2

def fun_3_der(x,):
    return np.array([2*x[0],2*x[1],2*x[2]])

'''
Define the gradient algorithm
'''

def gradient(function,x,dx):    #len(x)=dimension
    grad=np.zeros(len(x),)
    for i in range(len(x)):
        xx=np.zeros(len(x),)        #xx,xxx:  set all values of x to zero for the dimensions which are not considered in that iteration
        xx[i]=x[i]+dx
        xxx=np.zeros(len(x),)   
        xxx[i]=x[i]-dx
        gradient_v=(function(xx)-function(xxx))/2/dx    #The gradient definition according to the lecture slides week2
        grad[i]=gradient_v      #Writes the gradient of that iteration in the ith element of grad
    return grad 

'''
Input the variables here
'''  
    
def calculate_gradient():
    x=np.array([1.2,5,4])
    dx=0.1   
    grad_num=gradient(fun_3,x,dx)
    grad_ana=fun_3_der(x)
          
    if all (np.abs(grad_num-grad_ana)<0.001):
        print('Congratulations: gradient working')
    else:
        print('Gradient not working')

""" Run the programm """
def main():
    calculate_gradient()
    
if __name__=="__main__":
    main()
        