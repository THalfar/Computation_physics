# -*- coding: utf-8 -*-
"""
Exercise sheet 2 exercise 2: Integration

This script integrates over one dimension in a, over two in b and over 3 in c.

Created on Fri Jan 17 13:42:09 2020

@author: mvleko
"""
import numpy as np
from num_calculus import simpson_integration
from num_calculus import simpson_integration2

'''
Define the functions given on the exercise sheet
'''

def f_a1(x):
    return x**2*np.exp(-2*x)

def f_a2(x):
    return np.sin(x)/x

def f_a3(x):
    return np.exp(np.sin(x**3))

def f_b(x,y):
    return x*np.exp(-np.sqrt(x**2+y**2))

def f_c1(x,y,z,ra,rb):
    # ra and rb are 3dim lists
    integral=abs(np.exp(-np.sqrt((x-ra[0])**2+(y-ra[1])**2+(z-ra[2])**2))/np.sqrt(np.pi))**2 \
    /(np.sqrt((x-rb[0])**2+(y-rb[1])**2+(z-rb[2])**2))
    return integral

def f_c2(R):
    return (1-(1+R)*np.exp(-2*R))/R

'''
Define a function which ouputs the integration values
'''

def integrate_a1():
    x=np.linspace(0,5*10**4,100000)       #Linspace defines the integration limits
    eval_integral=simpson_integration(x,f_a1)
    print('Integral a1 =', eval_integral)

def integrate_a2():
    x=np.linspace(0.001,1,10000)        #Not from 0 to 1, because we divide through x
    eval_integral=simpson_integration(x,f_a2)
    print('Integral a2 =',eval_integral)

def integrate_a3():
    x=np.linspace(0,5,10000)
    eval_integral=simpson_integration(x,f_a3)
    print('Integral a3 =',eval_integral)

def integrate_b():
    x=np.linspace(0,2,100)
    y=np.linspace(-2,2,100)
    eval_integralx_list=np.zeros(len(x),)
    for i in range(len(y)):      #Loops through all y-values
        eval_integralx=0
        functionx=f_b(x,y[i])
        eval_integralx=simpson_integration2(x,functionx)
        eval_integralx_list[i]=eval_integralx
    eval_integral=simpson_integration2(y,eval_integralx_list)
        
    print('Integral b =',eval_integral)
    
def integrate_c(): 
    x=np.linspace(-5,5,100)     #x,y,z is the integration volume
    y=np.linspace(-5,5,100)
    z=np.linspace(-5,5,100)
    
    ra=np.array([0.7,0,0])      #r_a and r_b are arbitrary vectors
    rb=np.array([-0.7,0,0])
    
    eval_integraly_list = np.zeros(len(y),)   #Initialize the list, in which my partial integrations are stored
    eval_integralx_list = np.zeros(len(x),)
    
    for j in range(len(z)):             #Loops through all z-values
        for i in range(len(y)):         #Loops through all y-values
            functionx=f_c1(x,y[i],z[j],ra,rb) #f_c1 at one specific y and z value
            eval_integralx = simpson_integration2(x,functionx)
            eval_integralx_list[i] = eval_integralx
        eval_integraly_list[j] = simpson_integration2(y,eval_integralx_list)
    eval_integral = simpson_integration2(z,eval_integraly_list)    
        
    print('Integral c =',eval_integral)  
    
    eval_integral_ana = f_c2(np.sqrt((ra[0]-rb[0])**2+(ra[1]-rb[1])**2+(ra[2]-rb[2])**2))
    print('I_ana c',eval_integral_ana)



""" Run the program """
def main():
    integrate_a1()
    integrate_a2()
    integrate_a3()
    integrate_b()
    integrate_c()
    
if __name__=="__main__":
    main()