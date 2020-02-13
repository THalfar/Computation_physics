"""
Calculating some integrals using numeric methods.
"""

from scipy.integrate import simps
from scipy.integrate import tplquad
import numpy as np

# a)
# Function 1 approaches 0 quickly as x increases.
# Integrating to 100 is sufficient.
def fun1(r): return r**2*np.exp(-2*r)
x1 = np.linspace(0, 100, 100000)

# Function 2 does not go to infinity as x approaches 0,
# so integrating from very close to 0 to 1 gives accurate
# results (and converges to begin with).
def fun2(x): return np.sin(x)/x
x2 = np.linspace(0.00000000001,1,1000)

# Function 3 oscillates rapidly as x increases. A larger
# number of intervals is necessary.

def fun3(x): return np.exp(np.sin(x**3))
x3 = np.linspace(0,5,100000)


# b)
def fun4(x,y): return x*np.exp(-1*np.sqrt(x**2+y**2))
x4 = np.linspace(0,2,1000)
y4 = np.linspace(-2,2,1000)

#c)
def psii(x, y, z): return np.exp(-1*np.linalg.norm([x, y, z]))/np.sqrt(np.pi)
rA = [-1, 0, 0]
rB = [0, 1, 0]
def fun5(x, y, z):
    diffB = [x - rB[0], y - rB[1], z - rB[2]]
    return abs(psii(x-rA[0],y-rA[1],z-rA[2]))**2/(np.linalg.norm(diffB))

def main():

    # a)
    print("a)")
    print(simps(fun1(x1), x1))
    print(simps(fun2(x2), x2))
    print(simps(fun3(x3), x3))

    # b)
    print("")
    print("b)")
    ints = np.zeros(len(y4))
    # Calculate the integral dx for all points in y4, 'collapsing' the integral into one dimension.
    for i in range(len(y4)):
        ints[i] = simps(fun4(x4,y4[i]),x4)
    # calculate and print the integral of these collapsed values over y.
    print(simps(ints,y4))

    print("")
    print("c)")
    # a bad monte carlo integral
    I = 0
    for i in range(100000):
        x = np.random.rand() * 10 - 5
        y = np.random.rand() * 10 - 5
        z = np.random.rand() * 10 - 5
        I += fun5(x, y, z)
    print(I*10**3/100000)
    print("correct answer:")
    R = np.linalg.norm([rA[0] - rB[0], rA[1] - rB[1], rA[2] - rB[2]])
    print((1-(1+R)*np.exp(-2*R))/R)

main()