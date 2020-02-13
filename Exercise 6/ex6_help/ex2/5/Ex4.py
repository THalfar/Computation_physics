# -*- coding: utf-8 -*-
"""
This script finds the indices of the closest number of an arbitrary number in a linear or non linear array.
Excercise sheet 2, number 4

Created on Sun Jan 19 20:22:26 2020

@author: mvleko
"""

import numpy as np

'''
Define the index search algorithm with linear spacing
'''

def index(x,x_value):
    if x_value==x[0]:       #in case the value which is searched for is the first value of the array. DOESN'T WORK! I don't know why, it should work
        iteration=0
        return iteration
    elif x_value==x[-1]:    #in case the value is the last value
        iteration=-1
        return iteration
    elif x_value<x[0] or x_value>x[-1]: #in case the value is outside of the range: outputs false!
        iteration=False
        return iteration
    else:                   #in case the value is inside the range and not the first or last one
        xx=x[0]
        iteration=0
        while x_value>xx:   #add +1 to the iteration counter and go through every single value in x until the input value is not larger than the array element anymore
            iteration=iteration+1
            xx=x[iteration]
        return iteration
    
'''
Define a non-linear grid according to the exercise sheet
'''
    
def create_array(r0,rmax,dim):
    r=np.zeros(dim,)
    for i in range(1,dim):
        h=np.log(rmax/r0+1)/(dim-1)
        r[i]=r0*(np.exp(i*h)-1)
    r[0]=0.    
    return r
 
'''
Define a non-linear grid according to the exercise sheet
'''

def run():
    #linear case:
    x=np.linspace(-1,5,100)
    x_value=4      #The value which is searched for
    value_linear=index(x,x_value)
    if value_linear==False:
       print('Index for linear spacing not in x-range')
    else:
        print('Index for linear spacing: ',x_value ,'lies between ',x[value_linear-1],' and', x[value_linear])
        
    #Non linear case
    r0=1e-5
    rmax=100
    dim=99
    r=create_array(r0,rmax,dim)
    r_value=99      #The value which is searched for
    value=index(r,r_value)
    if value==False:
       print('Index for not-linear spacing not in x-range')
    else:
        print('Index for not-linear spacing: ',r_value ,'lies between ',r[value-1],' and', r[value])
    
    
    
    
""" Run the programm """
def main():
    run()
    
if __name__=="__main__":
    main()        