# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:41:35 2020

@author: halfar
"""
from numba import jit
from numpy import *
import matplotlib.pyplot as plt
import time

# papa = zeros((20,20), dtype = bool)

# print(papa[1,1])
# papa[2:7, 3] = True
# print(papa)

# plt.imshow(papa, cmap='hot', interpolation='nearest')

# @jit
def testi():
    
    papa = zeros((10000,10000))
    
    for i in range(papa.shape[0]):
        
        for j in range(papa.shape[1]):
            
            papa[i,j]  += i+j
    
    return papa

# @jit
def kaka():
    
    lala = []
    pau = 1000000
    
    for i in range(pau):
        
        lala.append(i**2)
     
    return lala
    

start_time = time.time()
tata = testi()
print ("did take ", time.time() - start_time, "s to run")

start_time = time.time()
papa = testi()
print ("did take ", time.time() - start_time, "s to run")

start_time = time.time()
tata = kaka()
print ("did take ", time.time() - start_time, "s to run")

start_time = time.time()
papa = kaka()
print ("did take ", time.time() - start_time, "s to run")
