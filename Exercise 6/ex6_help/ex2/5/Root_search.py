# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:22:26 2020

@author: mvleko
"""

import numpy as np

'''
Define test functions
'''

def fun(x):
    return x+1/2


'''
Define the root search algorithm with equal spacing
'''

def root_search(x,dx,tolerance):
    x_start=(x[-1]-x[0])/2
    x=np.zeros(3,)
    x[0]=x_start-dx
    x[1]=x_start
    x[2]=x[1]-(x[1]-x[0])*fun(x[1])/(fun(x[1])-fun(x[0]))
    iteration=0
    it_max=10000
    while abs(x[1]-x[0])<tolerance and iteration<it_max:
        xx=np.zeros(3,)
        xx=x
        xx[2]=xx[1]-(xx[1]-xx[0])*fun(xx[1])/(fun(xx[1])-fun(xx[0]))
        x[0]=xx[1]
        x[1]=xx[2]
        iteration=iteration+1
    print('Iteration', iteration)    
    return x


def run():
    x_input=np.linspace(-9,10,100)
    dx=0.1
    tolerance=0.1

    value=root_search(x_input,dx,tolerance)
    print('root @ ',value[1])
        

""" Run the programm """
def main():
    run()
    
if __name__=="__main__":
    main()    