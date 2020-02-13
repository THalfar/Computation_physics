"""
FYS-4096 problem 2 a) and b)
prints answers to the integrals
"""

import numpy as np
from scipy.integrate import simps, dblquad

def qa():
    # Uses numpy.simps to solve integrals
    x = np.linspace(0,1e4,20000)
    fun = lambda r: np.exp(-2*r)*r**2
    f = fun(x)
    ans = simps(f,x)
    print("Solution to first integral: ",ans)
    print("Absolute error: ", abs(0.25-ans))

    x = np.linspace(1e-6,1,250)
    fun = lambda x: np.sin(x)/x
    f = fun(x)
    ans = simps(f,x)
    print("Solution to second integral: ",ans)
    print("Absolute error: ", abs(0.25-ans))

    x = np.linspace(0, 5, 250)
    fun = lambda x: np.exp(np.sin(x**3))
    f = fun(x)
    ans = simps(f, x)
    print("Solution to third integral: ", ans)
    print("Absolute error: ", abs(0.25 - ans))

def qb():
    # Solves double integral with numpy.dblquad
    fun = lambda y,x: x*np.exp(-1*np.sqrt(x**2+y**2))
    I, err = dblquad(fun, 0,2,-2,2)
    print("Double integral answer: ",I)
    print("Error: ",err)

def qc():
    # Solves nothing, TODO
    psi = lambda r: np.exp(-1*np.linalg.norm(r))/np.sqrt(np.pi)
    ra = np.array([1,1,1])
    rb = np.array([1, -1, -1])
    


def main():
    qa()
    qb()

main()