""" 
-------- EXERCISE 1 - problem 4 ------------
----- FYS-4096 - Computational Physics -----

convergence_check.py plots the convergence of numerical
first and second derivative and simpson integral and 
saves the figure to convergenceN.pdf files. The functions
to be checked are imported from num_calculus.py. 

"""

import numpy as np
import matplotlib.pyplot as plt
from num_calculus import first_derivative
from num_calculus import second_derivative
from num_calculus import num_simpson


def conv_plot(fun, number, correct, fun_str):
    # fun      : function to be tested with
    # number   : number for names and titles
    # correct  : correct answer
    # fun_str  : function as a string
    
    # Takes a function and plots its convergence
    # to a figure and saves it with name and title
    # specified in funciton parameters
    
    # logarithmic values to check convergence
    dx = 10**(-np.linspace(1,5,21))
    
    value1 = np.zeros(21)
    value2 = np.zeros(21)
    value3 = np.zeros(21)
    
    # values for certain intervals
    for n in range(0,21):
        value1[n] = first_derivative(fun, 5, dx[n] ) - correct[0]
        value2[n] = second_derivative(fun, 5, dx[n] ) - correct[1]
        value3[n] = num_simpson(fun, 0, 5, int(2*round(1/dx[n]/2)) ) - correct[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # specify logaritmhic scale
    plt.yscale('log')
    plt.xscale('log')
    
    ax.plot(dx,abs(value1), label='first derivative', marker='o')
    ax.plot(dx,abs(value2), label='second derivative', marker='o')
    ax.plot(dx,abs(value3), label='simpson integral', marker='o')
    
    plt.legend(loc='upper right',
           ncol=1, borderaxespad=0.)
    
    # x axis limit and axis labels
    ax.set_xlim(dx.max(),dx.min())
    ax.set_xlabel('interval length')
    ax.set_ylabel('convergence')
    
    title = 'Convergence figure: ' + fun_str
    fig.suptitle(title)
    
    name = 'convergence' + str(number) + '.pdf'
    
    # save and show figures
    fig.savefig(name,dpi=200)
    plt.show()
    
    return

def main():
    
    # functions to be tested
    def fun1(x): return np.sin(2*x) * np.cos(x)
    def fun2(x): return np.exp(2*x) * np.cos(x)
    def fun3(x): return np.cos(2*x) * x**3 + 2*x
    def fun4(x): return np.sin(2*x**3) * x**2
    
    # functions as strings for titles
    fun_str1 = "sin(2x) * cos(x)"
    fun_str2 = "e^2x * cos(x)"
    fun_str3 = "cos(2x) * x^3 + 2x"
    fun_str4 = "sin(2x^3) * x^2"
    
    # corect answers
    correct1 = [-0.99770077656619, -2.446833143375457,  0.651450226078191]
    correct2 = [ 33617.8635864492,  103231.0772188675, -1725.512376981143]
    correct3 = [ 75.0749130416085,  557.5699519327436, -22.00417961153028]
    correct4 = [ 894.000864624309,  547004.5173100102,  0.126501949119143]
    
    # actual convergence tests
    conv_plot(fun1, 1, correct1, fun_str1)
    conv_plot(fun2, 2, correct2, fun_str2)
    conv_plot(fun3, 3, correct3, fun_str3)
    conv_plot(fun4, 4, correct4, fun_str4)
    
    return 0


if __name__=="__main__":
    main()
