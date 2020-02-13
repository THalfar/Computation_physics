"""
Linear interpolation in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

""" Edited for submission by Roosa Hytönen 255163
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def l1(t):
    """ Basis function for linear interpolation
    """
    return 1-t


def l2(t):
    """ Basis function for linear interpolation
        """
    return t


def test_function_1D():
    """ Test function for implementing linear interpolation in a 1-dimensional case
    """
    x = np.linspace(0., 2. * np.pi, 10000)
    y = np.sin(x)
    pts = [10, 20, 30, 40, 50]
    i = 0
    """ Varying the amount of grid points and calculating the average of two-norm of interpolated value and analytical 
        value
    """
    print("1D linear interpolation")
    print()
    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)
    """ Creating linear_interp object corresponding to 1-dimensional case
    """
    lin1d = linear_interp(x=x, f=y, dims=1)
    while i < 5:
        xx = np.linspace(0., 2. * np.pi, pts[i])
        """ Interpolating by calling the function eval1d for the class object lin1d
        """
        y_evaluated = lin1d.eval1d(xx)
        y_analytical = np.sin(xx)
        error = np.linalg.norm(y_evaluated-y_analytical)/pts[i]
        ax1d.plot(xx, y_evaluated)
        ax1d.set_xlabel(r'$x$')
        ax1d.set_ylabel(r'f(x)')
        ax1d.set_title('1D linear interpolation of sin(x) with varying number of grid points')
        fig1d.show()
        print("Average of two-norm with", pts[i], "grid points:", np.format_float_scientific(error,
                                                                                             unique=False,
                                                                                             precision=5))
        i += 1
    print()
    """ 1D case works logically, as with increasing grid point number the error decreases. As can be seen from the 
        figure, the line becomes smoother (making a legend in the loop is very difficult, but it is intuitive when 
        looking at the errors that the blue line corresponds to the smallest number of grid points
    """


def test_function_2D():
    """ Test function for implementing linear interpolation in a 2-dimensional case
    """
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 11)
    X, Y = np.meshgrid(x, y)
    Z = X * np.exp(-1.0 * (X * X + Y * Y))
    """ Number of intervals should be smaller, as the increasing number of dimensions increases computation time
    """
    pts = [10, 20, 30, 40, 50]
    i = 0
    """ Varying the amount of grid points and calculating the average of two-norm of interpolated value and analytical 
        value
    """
    print("2D linear interpolation")
    print()
    """ Creating linear_interp object corresponding to 2-dimensional case
    """
    lin2d = linear_interp(x=x, y=y, f=Z, dims=2)
    while i < 5:
        xx = np.linspace(-2.0, 2.0, pts[i])
        yy = np.linspace(-2.0, 2.0, pts[i])
        XX, YY = np.meshgrid(xx, yy)
        """ Interpolating by calling the function eval2d for the class object lin2d
        """
        Z_evaluated = lin2d.eval2d(xx, yy)
        Z_analytical = XX * np.exp(-1.0 * (XX * XX + YY * YY))
        error = np.linalg.norm(Z_evaluated - Z_analytical) / pts[i]
        print("Average of two-norm with", pts[i], "grid points:", np.format_float_scientific(error,
                                                                                             unique=False,
                                                                                             precision=5))
        i += 1
    print()
    """ 2D case is a bit unstable with varying grid point number
    """


def test_function_3D():
    """ Test function for implementing linear interpolation in a 3-dimensional case
    """
    x = np.linspace(0.0, 3.0, 10)
    y = np.linspace(0.0, 3.0, 10)
    z = np.linspace(0.0, 3.0, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    F = (X + Y + Z) * np.exp(-1.0 * (X * X + Y * Y + Z * Z))
    X, Y = np.meshgrid(x, y)
    """ Creating linear_interp object corresponding to 3-dimensional case
    """
    lin3d = linear_interp(x=x, y=y, z=z, f=F, dims=3)
    """ Number of intervals smaller, as the increased dimensionality increases computation time significantly
    """
    intervals = [10, 20, 30, 40, 50]
    i = 0
    """ Varying the amount of grid points and calculating the average of two-norm of interpolated value and analytical 
        value
    """
    print("3D linear interpolation")
    print()
    while i < 5:
        xx = np.linspace(0.0, 3.0, intervals[i])
        yy = np.linspace(0.0, 3.0, intervals[i])
        zz = np.linspace(0.0, 3.0, intervals[i])
        XX, YY, ZZ = np.meshgrid(xx, yy, zz)
        """ Interpolating by calling the function eval3d for the class object lin3d
        """
        F_evaluated = lin3d.eval3d(xx, yy, zz)
        F_analytical = (XX + YY + ZZ) * np.exp(-1.0 * (XX * XX + YY * YY + ZZ * ZZ))
        error = np.linalg.norm(F_evaluated - F_analytical) / intervals[i]
        print("Average of two-norm with", intervals[i], "grid points:", np.format_float_scientific(error,
                                                                                                   unique=False,
                                                                                                   precision=5))
        i += 1
    print()
    """ In the 3D case increasing the grid point number appears to decrease accuracy
    """


class linear_interp:
    def __init__(self, *args, **kwargs):
        """ Initialization of class with statements accounting for 1-3 dimensional objects
        """
        self.dims = kwargs['dims']
        if self.dims == 1:
            self.x = kwargs['x']
            self.f = kwargs['f']
            self.hx = np.diff(self.x)
        elif self.dims == 2:
            self.x = kwargs['x']
            self.y = kwargs['y']
            self.f = kwargs['f']
            self.hx = np.diff(self.x)
            self.hy = np.diff(self.y)
        elif self.dims == 3:
            self.x = kwargs['x']
            self.y = kwargs['y']
            self.z = kwargs['z']
            self.f = kwargs['f']
            self.hx = np.diff(self.x)
            self.hy = np.diff(self.y)
            self.hz = np.diff(self.z)
        else:
            print('Either dims is missing or specific dims is not available')
      
    def eval1d(self, x):
        """ Class function used to evaluate 1-dimensional linear interpolation.
        """
        if np.isscalar(x):
            x = np.array([x])
        N = len(self.x)-1
        f = np.zeros((len(x),))
        ii = 0
        """ Approximation of function values for multiple consecutive points contained in x
        """
        for val in x:
            i = np.floor(np.where(self.x <= val)[0][-1]).astype(int)
            """ if: the index is already at the last point
                else: interpolation according to definition in lecture material 
            """
            if i == N:
                f[ii] = self.f[i]
            else:
                t = (val-self.x[i])/self.hx[i]
                f[ii] = self.f[i]*l1(t)+self.f[i+1]*l2(t)
            ii += 1
        return f

    def eval2d(self, x, y):
        """ Class function used to evaluate 2-dimensional linear interpolation.
        """
        if np.isscalar(x):
            x = np.array([x])
        if np.isscalar(y):
            y = np.array([y])
        Nx = len(self.x)-1
        Ny = len(self.y)-1
        f = np.zeros((len(x), len(y)))
        A = np.zeros((2, 2))
        ii = 0
        """ Finding indices for x and y between which the evaluated point is by looping through x and y arrays
        """
        for valx in x:
            i = np.floor(np.where(self.x <= valx)[0][-1]).astype(int)
            if i == Nx:
                i -= 1
            jj = 0
            for valy in y:
                j = np.floor(np.where(self.y <= valy)[0][-1]).astype(int)
                if j == Ny:
                    j -= 1
                tx = (valx-self.x[i])/self.hx[i]
                ty = (valy-self.y[j])/self.hy[j]
                """ Forming arrays containing the defined basis functions 
                """
                ptx = np.array([l1(tx), l2(tx)])
                pty = np.array([l1(ty), l2(ty)])
                """ Forming the array containing the desired indices for evaluation
                """
                A[0, :] = np.array([self.f[i, j], self.f[i, j+1]])
                A[1, :] = np.array([self.f[i+1, j], self.f[i+1, j+1]])
                """ Interpolation according to definition in lecture material
                """
                f[ii, jj] = np.dot(ptx, np.dot(A, pty))
                jj += 1
            ii += 1
        return f
    # end eval2d

    def eval3d(self, x, y, z):
        """ Class function used to evaluate 3-dimensional linear interpolation.
        """
        if np.isscalar(x):
            x = np.array([x])
        if np.isscalar(y):
            y = np.array([y])
        if np.isscalar(z):
            z = np.array([z])
        Nx = len(self.x)-1
        Ny = len(self.y)-1
        Nz = len(self.z)-1
        f = np.zeros((len(x), len(y), len(z)))
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        ii = 0
        """ Finding indices for x, y and z between which the evaluated point is by looping through x, y and z arrays
        """
        for valx in x:
            i = np.floor(np.where(self.x <= valx)[0][-1]).astype(int)
            if i == Nx:
                i -= 1
            jj = 0
            for valy in y:
                j = np.floor(np.where(self.y <= valy)[0][-1]).astype(int)
                if j == Ny:
                    j -= 1
                kk = 0
                for valz in z:
                    k = np.floor(np.where(self.z <= valz)[0][-1]).astype(int)
                    if k == Nz:
                        k -= 1
                    tx = (valx-self.x[i])/self.hx[i]
                    ty = (valy-self.y[j])/self.hy[j]
                    tz = (valz-self.z[k])/self.hz[k]
                    """ Forming arrays containing the defined basis functions 
                    """
                    ptx = np.array([l1(tx), l2(tx)])
                    pty = np.array([l1(ty), l2(ty)])
                    ptz = np.array([l1(tz), l2(tz)])
                    """ Forming the array A containing the desired indices for evaluation using array B for help
                    """
                    B[0, :] = np.array([self.f[i, j, k], self.f[i, j, k+1]])
                    B[1, :] = np.array([self.f[i+1, j, k], self.f[i+1, j, k+1]])
                    A[:, 0] = np.dot(B, ptz)
                    B[0, :] = np.array([self.f[i, j+1, k], self.f[i, j+1, k+1]])
                    B[1, :] = np.array([self.f[i+1, j+1, k], self.f[i+1, j+1, k+1]])
                    A[:, 1] = np.dot(B, ptz)
                    """ Linear interpolation according to definition in lecture material
                    """
                    f[ii, jj, kk] = np.dot(ptx, np.dot(A, pty))
                    kk += 1
                jj += 1
            ii += 1
        return f
    # end eval3d
# end class linear interp

    
def main():
    test_function_1D()
    test_function_2D()
    test_function_3D()

    # fig1d = plt.figure()
    # ax1d = fig1d.add_subplot(111)
    # 1d example
    # x = np.linspace(0., 2.*np.pi, 10)
    # y = np.sin(x)
    # lin1d = linear_interp(x=x, f=y, dims=1)
    # xx = np.linspace(0., 2.*np.pi, 100)
    # ax1d.plot(xx, lin1d.eval1d(xx))
    # ax1d.plot(x, y, 'o', xx, np.sin(xx), 'r--')
    # ax1d.set_title('function')

    # 2d example
    # fig2d = plt.figure()
    # ax2d = fig2d.add_subplot(221, projection='3d')
    # ax2d2 = fig2d.add_subplot(222, projection='3d')
    # ax2d3 = fig2d.add_subplot(223)
    # ax2d4 = fig2d.add_subplot(224)
    # x = np.linspace(-2.0, 2.0, 11)
    # y = np.linspace(-2.0, 2.0, 11)
    # X, Y = np.meshgrid(x, y)
    # Z = X*np.exp(-1.0*(X*X+Y*Y))
    # ax2d.plot_wireframe(X, Y, Z)
    # ax2d3.pcolor(X, Y, Z)
    # ax2d3.contourf(X,Y,Z)

    # lin2d = linear_interp(x=x, y=y, f=Z, dims=2)
    # x = np.linspace(-2.0, 2.0, 51)
    # y = np.linspace(-2.0, 2.0, 51)
    # X, Y = np.meshgrid(x, y)
    # Z = lin2d.eval2d(x, y)
    # ax2d2.plot_wireframe(X, Y, Z)
    # ax2d4.pcolor(X, Y, Z)
    
    # 3d example
    # x = np.linspace(0.0, 3.0, 10)
    # y = np.linspace(0.0, 3.0, 10)
    # z = np.linspace(0.0, 3.0, 10)
    # X, Y, Z = np.meshgrid(x, y, z)
    # F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    # X, Y = np.meshgrid(x, y)
    # fig3d = plt.figure()
    # ax = fig3d.add_subplot(121)
    # ax.pcolor(X, Y, F[..., int(len(z)/2)])
    # lin3d = linear_interp(x=x, y=y, z=z, f=F, dims=3)
    
    # x = np.linspace(0.0, 3.0, 50)
    # y = np.linspace(0.0, 3.0, 50)
    # z = np.linspace(0.0, 3.0, 50)
    # X, Y = np.meshgrid(x, y)
    # F = lin3d.eval3d(x, y, z)
    # ax2 = fig3d.add_subplot(122)
    # ax2.pcolor(X, Y, F[..., int(len(z)/2)])

    plt.show()
# end main

    
if __name__ == "__main__":
    main()
