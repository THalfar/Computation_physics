# Computational Physics FYS-4096, Spring 2020
# Exercise 2, problem 2
#
# Pauli Huusari, 234214
# pauli.huusari@tuni.fi

import numpy as np
from num_calculus import trapezoid

# Sub-section (a) begins.

def first_function(x):
   return (x**2)*(np.exp(-2*x))

def second_function(x):
   return (np.sin(x))/x

def third_function(x):
   return np.exp(np.sin(x**3))

def first_integral():
   print('----- First integral begins. -----')
   interval = np.linspace(0,0.5,2)
   integral_value = trapezoid(interval,first_function)
   print('1 estimate is:',integral_value)
   comparison_value = 1
   i = 2
   while abs(comparison_value - integral_value) >= 10**(-6):
      interval = np.linspace(0,np.size(interval)+2,np.size(interval)*8)
      comparison_value = integral_value
      integral_value = trapezoid(interval,first_function)
      print(i,'estimate is:',integral_value)
      i = i + 1
   print('Final grid is: [',interval[0],',', interval[-1],'].')
   print('There are',np.size(interval),'grid points.')
   print('Final value is:', round(integral_value,6),'.')
   print('----- First integral ends. -----')
   return
   
def second_integral():
   print('----- Second integral begins. -----')
   interval = np.linspace(0.01,1,5)
   integral_value = trapezoid(interval,second_function)
   print('1 estimate is:',integral_value)
   comparison_value = 1
   i = 2
   while abs(comparison_value - integral_value) >= 10**(-6):
      interval = np.linspace(interval[0]/10,1,np.size(interval)*6)
      comparison_value = integral_value
      integral_value = trapezoid(interval,second_function)
      print(i,'estimate is:',integral_value)
      i = i + 1
   print('Final grid is: [',interval[0],',', interval[-1],'].')
   print('There are',np.size(interval),'grid points.')
   print('Final value is:', round(integral_value,6),'.')
   print('----- Second integral ends. -----')
   return

def third_integral():
   print('----- Third integral begins. -----')
   interval = np.linspace(0.0, 5.0, 10)
   integral_value = trapezoid(interval,third_function)
   print('1 estimate is:',integral_value)
   comparison_value = 1
   i = 2
   while abs(comparison_value - integral_value) >= 10**(-6):
      interval = np.linspace(0.0,5.0,np.size(interval)*6)
      comparison_value = integral_value
      integral_value = trapezoid(interval,third_function)
      print(i,'estimate is:',integral_value)
      i = i + 1
   print('Final grid is: [',interval[0],',', interval[-1],'].')
   print('There are',np.size(interval),'grid points.')
   print('Final value is:', round(integral_value,6),'.')
   print('----- Third integral ends. -----')
   return

# Sub-section (a) ends.
# Sub-section (b) begins.
# Implemented using Riemann-sum.

def fourth_function(x,y):
   return x*(np.exp(-1*(np.sqrt(x**2+y**2))))

def fourth_integral():
   print('----- Fourth integral begins. -----')
   x_interval = np.linspace(0.0, 2.0, 100)
   y_interval = np.linspace(-2.0, 2.0, 100)
   x_interval = x_interval[:-1]
   y_interval = y_interval[:-1]
   X, Y = np.meshgrid(x_interval, y_interval)
   dxdy = (x_interval[1]-x_interval[0])*(y_interval[1]-y_interval[0])
   integral_value = np.sum(fourth_function(X, Y))*dxdy
   print('1 estimate is:',integral_value)
   comparison_value = 1
   i = 2
   while abs(comparison_value - integral_value) >= 10**(-4):
      x_interval = np.linspace(0.0, 2.0, np.size(x_interval)*3)
      y_interval = np.linspace(-2.0, 2.0, np.size(y_interval)*2)
      x_interval = x_interval[:-1]
      y_interval = y_interval[:-1]
      X, Y = np.meshgrid(x_interval, y_interval)
      dxdy = (x_interval[1]-x_interval[0])*(y_interval[1]-y_interval[0])
      comparison_value = integral_value
      integral_value = np.sum(fourth_function(X, Y))*dxdy
      print(i,'estimate is:',integral_value)
      i = i + 1
   print('Final grid is: [',x_interval[0],',', x_interval[-1],']x[',y_interval[0],',',y_interval[-1],'].')
   print('There are',np.size(x_interval)-1,'x-grid points and',np.size(y_interval)-1,'y-grid points.')
   print('Total number of grid points is:',(np.size(x_interval)-1)*np.size(y_interval)-1,'.')
   print('Final value is:', round(integral_value,6),'.')
   print('----- Fourth integral ends. -----')
   return

# Sub-section (b) ends.
# Sub-section (c) begins.

def norm(point_1,point_2):
   '''
   NORM returns the Euclidean-norm of points point_1 and point_2.
   NORM is the distance between points point_1 and point_2.

   Point_1 and point_2 are three-element arrays.
   '''
   return np.sqrt((point_1[0]-point_2[0])**2+(point_1[1]-point_2[1])**2+(point_1[2]-point_2[2])**2)

def fifth_function(random_point_x,random_point_y,random_point_z,point_A, point_B):
   point_x=[random_point_x,random_point_y,random_point_z]
   return (np.exp(-2*norm(point_x, point_A)))/(np.pi*norm(point_x, point_B))

def fifth_function_analytic(point_A, point_B):
   R = norm(point_A, point_B)
   return (1 - (1 + R) * (np.exp(-2 * R)))/R

def fifth_integral(point_A, point_B):
   print('----- Point A is', point_A,'and point B is:',point_B)
   initial_interval_size = np.amax([abs(point_A), abs(point_B)])
   x_interval = np.linspace(-1*initial_interval_size, initial_interval_size, 10)
   y_interval = np.linspace(-1*initial_interval_size, initial_interval_size, 10)
   z_interval = np.linspace(-1*initial_interval_size, initial_interval_size, 10)
   
   X, Y, Z = np.meshgrid(x_interval, y_interval, z_interval)
   dxdydz = (x_interval[1]-x_interval[0])*(y_interval[1]-y_interval[0])*(z_interval[1]-z_interval[0])
   integral_value = 0.0
   integral_value = np.sum(fifth_function(X, Y, Z, point_A, point_B))*dxdydz
   print('1 estimate is:',integral_value)
   comparison_value = 1
   i = 2
   while abs(comparison_value - integral_value) >= 10**(-5) and i<26:
      x_interval = np.linspace(x_interval[0]-1, x_interval[-1]+1, np.size(x_interval)+12)
      y_interval = np.linspace(y_interval[0]-1, y_interval[-1]+1, np.size(y_interval)+12)
      z_interval = np.linspace(z_interval[0]-1, z_interval[-1]+1, np.size(z_interval)+12)
   
      X, Y, Z = np.meshgrid(x_interval, y_interval, z_interval)
      dxdydz = (x_interval[1]-x_interval[0])*(y_interval[1]-y_interval[0])*(z_interval[1]-z_interval[0])
      
      comparison_value = integral_value
      integral_value = np.sum(fifth_function(X, Y, Z, point_A, point_B))*dxdydz
      print(i,'estimate is:',integral_value)
      i = i + 1

   print('Final value is:', round(integral_value,6),'.')
   print('Analytic value is:',fifth_function_analytic(point_A, point_B))
   if i == 26:
      print('Calculating stopped after 25 iterations.')
   print('-----')
   return

def main():
   first_integral()
   second_integral()
   third_integral()
   fourth_integral()

   print('----- Fifth integral begins. -----')
   fifth_function_points_A = np.array([[1,1.5,-2],[0.3,0.5,-1.1],[1,1,1],[0,0,0],[-2,-3,-1.5]])
   fifth_function_points_B = np.array([[0.7,-0.3,2.1],[-0.5,0.3,1.2],[1.5,1.7,1.1],[-2.4,-3.1,-1.3],[4.2,3.6,3.3]])
   for point_A in fifth_function_points_A:
      for point_B in fifth_function_points_B:
         fifth_integral(point_A,point_B)
   print('----- Fifth integral ends. -----')

if __name__=="__main__":
   main()
