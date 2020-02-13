"""
Cubic hermite splines in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

from numpy import *
from matplotlib.pyplot import *


# basis functions from lecture slides
def p1(t): return (1 + 2 * t) * (t - 1) ** 2


def p2(t): return (t ** 2) * (3 - 2 * t)


def q1(t): return t * (t - 1) ** 2


def q2(t): return (t ** 2) * (t - 1)


def init_1d_spline(x, f, h):
    # now using complete boundary conditions
    # with forward/backward derivative
    # - natural boundary conditions commented
    a = zeros((len(x),))
    b = zeros((len(x),))
    c = zeros((len(x),))
    d = zeros((len(x),))
    fx = zeros((len(x),))

    # a[0]=1.0 # not needed
    b[0] = 1.0

    # natural boundary conditions 
    # c[0]=0.5
    # d[0]=1.5*(f[1]-f[0])/(x[1]-x[0])

    # complete boundary conditions
    c[0] = 0.0
    d[0] = (f[1] - f[0]) / (x[1] - x[0])

    for i in range(1, len(x) - 1):
        d[i] = 6.0 * (h[i] / h[i - 1] - h[i - 1] / h[i]) * f[i] - 6.0 * h[i] / h[i - 1] * f[i - 1] + 6.0 * h[i - 1] / h[
            i] * f[i + 1]
        a[i] = 2.0 * h[i]
        b[i] = 4.0 * (h[i] + h[i - 1])
        c[i] = 2.0 * h[i - 1]
        # end for

    b[-1] = 1.0
    # c[-1]=1.0 # not needed

    # natural boundary conditions
    # a[-1]=0.5
    # d[-1]=1.5*(f[-1]-f[-2])/(x[-1]-x[-2])

    # complete boundary conditions
    a[-1] = 0.0
    d[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])

    # solve tridiagonal eq. A*f=d
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]
    for i in range(1, len(x) - 1):
        temp = b[i] - c[i - 1] * a[i]
        c[i] = c[i] / temp
        d[i] = (d[i] - d[i - 1] * a[i]) / temp
    # end for

    fx[-1] = d[-1]
    for i in range(len(x) - 2, -1, -1):
        fx[i] = d[i] - c[i] * fx[i + 1]
    # end for

    return fx


# end function init_1d_spline


class spline:

    def __init__(self, *args, **kwargs):
        """
        Constructor sets the 1D, 2D or 3D grid to the object.

        :param args:
        :param kwargs: x, y, z are the grid points in x, y, and z direction (numpy arrays)
                       dims is the number of dimensions
                       f is the function values at grid points (1D, 2D or 3D numpy matrices)

        """
        self.dims = kwargs['dims']
        if (self.dims == 1):
            self.x = kwargs['x']  # grid in x direction
            self.f = kwargs['f']  # function values at grid points
            self.hx = diff(self.x)  # intervals (don't have to be uniform)
            self.fx = init_1d_spline(self.x, self.f, self.hx)
        elif (self.dims == 2):
            self.x = kwargs['x']
            self.y = kwargs['y']
            self.f = kwargs['f']
            self.hx = diff(self.x)
            self.hy = diff(self.y)
            self.fx = zeros(shape(self.f))
            self.fy = zeros(shape(self.f))
            self.fxy = zeros(shape(self.f))
            for i in range(max([len(self.x), len(self.y)])):
                # initiate 2d spline by initiating 1d splines in both x and y direction
                if (i < len(self.y)):
                    self.fx[:, i] = init_1d_spline(self.x, self.f[:, i], self.hx)
                if (i < len(self.x)):
                    self.fy[i, :] = init_1d_spline(self.y, self.f[i, :], self.hy)
            # end for
            for i in range(len(self.y)):
                self.fxy[:, i] = init_1d_spline(self.x, self.fy[:, i], self.hx)
            # end for
        elif (self.dims == 3):
            self.x = kwargs['x']
            self.y = kwargs['y']
            self.z = kwargs['z']
            self.f = kwargs['f']
            self.hx = diff(self.x)
            self.hy = diff(self.y)
            self.hz = diff(self.z)
            self.fx = zeros(shape(self.f))
            self.fy = zeros(shape(self.f))
            self.fz = zeros(shape(self.f))
            self.fxy = zeros(shape(self.f))
            self.fxz = zeros(shape(self.f))
            self.fyz = zeros(shape(self.f))
            self.fxyz = zeros(shape(self.f))
            for i in range(max([len(self.x), len(self.y), len(self.z)])):
                for j in range(max([len(self.x), len(self.y), len(self.z)])):
                    if (i < len(self.y) and j < len(self.z)):
                        self.fx[:, i, j] = init_1d_spline(self.x, self.f[:, i, j], self.hx)
                    if (i < len(self.x) and j < len(self.z)):
                        self.fy[i, :, j] = init_1d_spline(self.y, self.f[i, :, j], self.hy)
                    if (i < len(self.x) and j < len(self.y)):
                        self.fz[i, j, :] = init_1d_spline(self.z, self.f[i, j, :], self.hz)
            # end for
            for i in range(max([len(self.x), len(self.y), len(self.z)])):
                for j in range(max([len(self.x), len(self.y), len(self.z)])):
                    if (i < len(self.y) and j < len(self.z)):
                        self.fxy[:, i, j] = init_1d_spline(self.x, self.fy[:, i, j], self.hx)
                    if (i < len(self.y) and j < len(self.z)):
                        self.fxz[:, i, j] = init_1d_spline(self.x, self.fz[:, i, j], self.hx)
                    if (i < len(self.x) and j < len(self.z)):
                        self.fyz[i, :, j] = init_1d_spline(self.y, self.fz[i, :, j], self.hy)
            # end for
            for i in range(len(self.y)):
                for j in range(len(self.z)):
                    self.fxyz[:, i, j] = init_1d_spline(self.x, self.fyz[:, i, j], self.hx)
            # end for
        else:
            print('Either dims is missing or specific dims is not available')
        # end if

    def eval1d(self, x):
        """
        Evaluates function values at points in x using Hermite cubic splines (1D).

        :param x: numpy array of points for which interpolation is calculated
        :return: f: interpolated function values corresponding to x
        """
        if isscalar(x):
            x = array([x])
        N = len(self.x) - 1
        f = zeros((len(x),))
        ii = 0
        for val in x:
            i = floor(where(self.x <= val)[0][-1]).astype(int)
            if i == N:
                f[ii] = self.f[i]
            else:
                t = (val - self.x[i]) / self.hx[i]
                f[ii] = self.f[i] * p1(t) + self.f[i + 1] * p2(t) + self.hx[i] * (
                            self.fx[i] * q1(t) + self.fx[i + 1] * q2(t))
            ii += 1

        return f

    # end eval1d

    def eval2d(self, x, y):
        """
        Evaluates function values at points in (x,y) using Hermite cubic splines (2D).

        :param x: numpy array of points for which interpolation is calculated
        :param y: numpy array of points for which interpolation is calculated
        :return: f: function value matrix corresponding to x and y
        """
        if isscalar(x):
            x = array([x])
        if isscalar(y):
            y = array([y])
        Nx = len(self.x) - 1
        Ny = len(self.y) - 1
        f = zeros((len(x), len(y)))
        A = zeros((4, 4))
        ii = 0
        for valx in x:
            i = floor(where(self.x <= valx)[0][-1]).astype(int)
            if (i == Nx):
                i -= 1
            jj = 0
            for valy in y:
                j = floor(where(self.y <= valy)[0][-1]).astype(int)
                if (j == Ny):
                    j -= 1

                # using formulas from lecture slides
                u = (valx - self.x[i]) / self.hx[i]
                v = (valy - self.y[j]) / self.hy[j]
                pu = array([p1(u), p2(u), self.hx[i] * q1(u), self.hx[i] * q2(u)])
                pv = array([p1(v), p2(v), self.hy[j] * q1(v), self.hy[j] * q2(v)])
                A[0, :] = array([self.f[i, j], self.f[i, j + 1], self.fy[i, j], self.fy[i, j + 1]])
                A[1, :] = array([self.f[i + 1, j], self.f[i + 1, j + 1], self.fy[i + 1, j], self.fy[i + 1, j + 1]])
                A[2, :] = array([self.fx[i, j], self.fx[i, j + 1], self.fxy[i, j], self.fxy[i, j + 1]])
                A[3, :] = array([self.fx[i + 1, j], self.fx[i + 1, j + 1], self.fxy[i + 1, j], self.fxy[i + 1, j + 1]])

                f[ii, jj] = dot(pu, dot(A, pv))
                jj += 1
            ii += 1
        return f

    # end eval2d

    def eval3d(self, x, y, z):
        """
        Evaluates function values at points in (x,y,z) using Hermite cubic splines (3D).

        :param x: numpy array of points for which interpolation is calculated
        :param y: numpy array of points for which interpolation is calculated
        :param z: numpy array of points for which interpolation is calculated
        :return: f: function value matrix at points (x,y,z)
        """
        if isscalar(x):
            x = array([x])
        if isscalar(y):
            y = array([y])
        if isscalar(z):
            z = array([z])
        Nx = len(self.x) - 1
        Ny = len(self.y) - 1
        Nz = len(self.z) - 1
        f = zeros((len(x), len(y), len(z)))
        A = zeros((4, 4))
        B = zeros((4, 4))
        ii = 0
        for valx in x:
            i = floor(where(self.x <= valx)[0][-1]).astype(int)
            if (i == Nx):
                i -= 1
            jj = 0
            for valy in y:
                j = floor(where(self.y <= valy)[0][-1]).astype(int)
                if (j == Ny):
                    j -= 1
                kk = 0
                for valz in z:

                    k = floor(where(self.z <= valz)[0][-1]).astype(int)
                    if (k == Nz):
                        k -= 1

                    # using formulas from lecture slides
                    u = (valx - self.x[i]) / self.hx[i]
                    v = (valy - self.y[j]) / self.hy[j]
                    t = (valz - self.z[k]) / self.hz[k]
                    pu = array([p1(u), p2(u), self.hx[i] * q1(u), self.hx[i] * q2(u)])
                    pv = array([p1(v), p2(v), self.hy[j] * q1(v), self.hy[j] * q2(v)])
                    pt = array([p1(t), p2(t), self.hz[k] * q1(t), self.hz[k] * q2(t)])
                    B[0, :] = array([self.f[i, j, k], self.f[i, j, k + 1], self.fz[i, j, k], self.fz[i, j, k + 1]])
                    B[1, :] = array(
                        [self.f[i + 1, j, k], self.f[i + 1, j, k + 1], self.fz[i + 1, j, k], self.fz[i + 1, j, k + 1]])
                    B[2, :] = array([self.fx[i, j, k], self.fx[i, j, k + 1], self.fxz[i, j, k], self.fxz[i, j, k + 1]])
                    B[3, :] = array([self.fx[i + 1, j, k], self.fx[i + 1, j, k + 1], self.fxz[i + 1, j, k],
                                     self.fxz[i + 1, j, k + 1]])
                    A[:, 0] = dot(B, pt)
                    B[0, :] = array(
                        [self.f[i, j + 1, k], self.f[i, j + 1, k + 1], self.fz[i, j + 1, k], self.fz[i, j + 1, k + 1]])
                    B[1, :] = array([self.f[i + 1, j + 1, k], self.f[i + 1, j + 1, k + 1], self.fz[i + 1, j + 1, k],
                                     self.fz[i + 1, j + 1, k + 1]])
                    B[2, :] = array([self.fx[i, j + 1, k], self.fx[i, j + 1, k + 1], self.fxz[i, j + 1, k],
                                     self.fxz[i, j + 1, k + 1]])
                    B[3, :] = array([self.fx[i + 1, j + 1, k], self.fx[i + 1, j + 1, k + 1], self.fxz[i + 1, j + 1, k],
                                     self.fxz[i + 1, j + 1, k + 1]])
                    A[:, 1] = dot(B, pt)

                    B[0, :] = array([self.fy[i, j, k], self.fy[i, j, k + 1], self.fyz[i, j, k], self.fyz[i, j, k + 1]])
                    B[1, :] = array([self.fy[i + 1, j, k], self.fy[i + 1, j, k + 1], self.fyz[i + 1, j, k],
                                     self.fyz[i + 1, j, k + 1]])
                    B[2, :] = array(
                        [self.fxy[i, j, k], self.fxy[i, j, k + 1], self.fxyz[i, j, k], self.fxyz[i, j, k + 1]])
                    B[3, :] = array([self.fxy[i + 1, j, k], self.fxy[i + 1, j, k + 1], self.fxyz[i + 1, j, k],
                                     self.fxyz[i + 1, j, k + 1]])
                    A[:, 2] = dot(B, pt)
                    B[0, :] = array([self.fy[i, j + 1, k], self.fy[i, j + 1, k + 1], self.fyz[i, j + 1, k],
                                     self.fyz[i, j + 1, k + 1]])
                    B[1, :] = array([self.fy[i + 1, j + 1, k], self.fy[i + 1, j + 1, k + 1], self.fyz[i + 1, j + 1, k],
                                     self.fyz[i + 1, j + 1, k + 1]])
                    B[2, :] = array([self.fxy[i, j + 1, k], self.fxy[i, j + 1, k + 1], self.fxyz[i, j + 1, k],
                                     self.fxyz[i, j + 1, k + 1]])
                    B[3, :] = array(
                        [self.fxy[i + 1, j + 1, k], self.fxy[i + 1, j + 1, k + 1], self.fxyz[i + 1, j + 1, k],
                         self.fxyz[i + 1, j + 1, k + 1]])
                    A[:, 3] = dot(B, pt)

                    f[ii, jj, kk] = dot(pu, dot(A, pv))
                    kk += 1
                jj += 1
            ii += 1
        return f
    # end eval3d


# end class spline

def test_1d():
    """
    function tests eval1d interpolation with a simple x^2 function through range [1,9]
    grid interval is 0.5, interpolated interval is 0.01 (50 times more dense)
    """
    x = arange(1, 10, 0.5)
    y = x ** 2
    spline_1d = spline(x=x, f=y, dims=1)
    xx = arange(1, 9, 0.01)

    interp_values = spline_1d.eval1d(xx)
    analytical_values = xx ** 2

    rel_err_array = 100 * abs(1 - interp_values / analytical_values)  # relative error array in percentage
    avg_rel_err = mean(rel_err_array)
    print("SPLINE 1D INTERPOLATION TEST______________________________________")
    print("Average relative error for 1d test case is: " + str(avg_rel_err) + " %")
    print("Function used: x^2")
    print("Grid used: 1:0.5:10")
    print("Interpolated to grid: 1:0.01:9\n")


def test_2d():
    """
    function tests eval1d interpolation with a simple x^2*y^2 function through range x=[1:9] and y=[1:9]
    grid interval is 0.5, interpolated interval is 0.1 (5 times more dense)
    """
    x = arange(0, 10, 0.5)
    y = x
    X, Y = meshgrid(x, y)
    Z = X ** 2 * Y ** 2
    spline_2d = spline(x=x, y=y, f=Z, dims=2)
    xx = arange(1, 9, 0.1)
    yy = xx

    interp_values = spline_2d.eval2d(xx, yy)
    XX, YY = meshgrid(xx, yy)
    analytical_values = XX ** 2 * YY ** 2

    rel_err_array = 100 * abs(1 - interp_values / analytical_values)  # relative error array in percentage
    avg_rel_err = mean(rel_err_array)
    print("SPLINE 2D INTERPOLATION TEST______________________________________")
    print("Average relative error for 2d test case is: " + str(avg_rel_err) + " %")
    print("Function used: x^2*y^2")
    print("Grid used: 1:0.5:10 (for both x and y)")
    print("Interpolated to grid: 1:0.1:9 (for both x and y)\n")


def test_3d():
    """
    function tests eval1d interpolation with a simple x^2*y^2*z^2 function through range x,y,z=[1:3]
    grid interval is 0.5, interpolated interval is 0.1 (5 times more dense)
    """
    x = arange(0, 3, 0.5)
    y = x
    z = x
    X, Y, Z = meshgrid(x, y, z)
    F = X ** 2 * Y ** 2 * Z ** 2
    spline_3d = spline(x=x, y=y, z=z, f=F, dims=3)
    xx = arange(1, 2.9, 0.1)
    yy = xx
    zz = xx

    interp_values = spline_3d.eval3d(xx, yy, zz)
    XX, YY, ZZ = meshgrid(xx, yy, zz)
    analytical_values = XX ** 2 * YY ** 2 * ZZ ** 2

    rel_err_array = 100 * abs(1 - interp_values / analytical_values)  # relative error array in percentage
    avg_rel_err = mean(rel_err_array)
    print("SPLINE 3D INTERPOLATION TEST______________________________________")
    print("Average relative error for 3d test case is: " + str(avg_rel_err) + " %")
    print("Function used: x^2*y^2*z^2")
    print("Grid used: 1:0.5:3 (for x, y and z)")
    print("Interpolated to grid: 1:0.1:3 (for x, y and z)\n")


def run_tests():
    test_1d()
    test_2d()
    test_3d()


def smooth1d(x, f, factor=3):
    """
        Function first averages the function values to a less dense grid.
        Then these averaged values are used to interpolate the function to the original input grid.
        The result is a smoother function

    :param x: grid points
    :param f: function values at grid points
    :param factor: smoothing factor, how many adjacent values are used for averaging
    :return: smoothed function values corresponding to grid points in x
    """

    xx = []  # less dense grid, density determinated by argument factor
    ff = []  # averaged values corresponding to xx gridpoints

    xx.append(x[0])  # add first point separately
    ff.append(mean(f[0:factor]))  # average of function values (only right side values available)

    i = 1
    while (i + 1) * factor < len(x):
        xx.append(x[i * factor])  # less dense grid, according to factor
        ff.append(mean(f[(i - 1) * factor: (i + 1) * factor]))  # average of left side and right side values

        i += 1

    xx.append(x[x.shape[0] - 1])  # add last point separately
    ff.append(mean(f[-1 - factor:-1]))  # only left side values are available for averaging

    # interpolate averaged function to the original grid and return it
    xx = asarray(xx)
    ff = asarray(ff)
    spline_1d = spline(x=xx, f=ff, dims=1)
    interp_values = spline_1d.eval1d(x)

    return interp_values


def smooth2d(x, y, f, factor=3):
    """
        simple 2d smoothing function which uses the smooth1d function in both x and y directions and averages them
        then returns the smoothed function values

    :param x: x grid
    :param y: y grid
    :param f: function values at (x, y)
    :param factor: smoothing factor, how many adjacent values are used for averaging
    :return: smoothed function values corresponding to grid points in (x, y)
    """

    # empty arrays for function values smoothed by x-direction and y-direction
    smoothed_f_by_x = np.empty([x.shape[0], y.shape[0]])
    smoothed_f_by_y = np.empty([x.shape[0], y.shape[0]])

    # smooth each x row in one dimension using smooth1d function
    for y_idx in range(y.shape[0]):
        fun_1d_x = f[:, y_idx] # get one dimensional function values (values corresponding to one row of x-coordinates)

        # use smooth1d function to smooth the one-dimensional function values. Add smoothed func values to array.
        smoothed_f_by_x[:,y_idx] = smooth1d(x[0,:], fun_1d_x, factor)

    # exactly the same method in y direction
    for x_idx in range(x.shape[0]):
        fun_1d_y = f[x_idx, :]
        smoothed_f_by_y[x_idx, :] = smooth1d(y[:,0], fun_1d_y, factor)

    # actual smoothed function is the average of the
    # functions smoothed in terms of x and function smoothed in terms of y
    average_smoothed_f = np.mean([smoothed_f_by_x, smoothed_f_by_y], axis=0)

    return average_smoothed_f


def smooth3d(x, y, z, f, factor=3):
    ...
    ...
    ...
    return ...


def test_smooth1d():

    # test smoothing of sin(x) with some random noise.
    x = linspace(0., 2. * pi, 500)
    y = sin(x)

    # loop to generate random noise to function
    for i in range(y.shape[0]):
        noise = 0.4 * (random.rand() - 0.5)  # random number between [-0.1, 0.1]
        y[i] += noise

    subplots(nrows=2, ncols=2)
    suptitle('Test of 1D smoothing. Function used: sin(x), with random noise.')

    subplot(2, 2, 1)
    plot(x, y, 'r-')
    plot(x, smooth1d(x, y, 5), 'b-', lw=4)
    xlim([0, 2 * pi])
    title("Smoothing with factor 5")

    subplot(2, 2, 2)
    plot(x, y, 'r-')
    plot(x, smooth1d(x, y, 10), 'b-', lw=4)
    xlim([0, 2 * pi])
    title("Smoothing with factor 10")

    subplot(2, 2, 3)
    plot(x, y, 'r-')
    plot(x, smooth1d(x, y, 20), 'b-', lw=4)
    xlim([0, 2 * pi])
    title("Smoothing with factor 20")

    subplot(2, 2, 4)
    plot(x, y, 'r-')
    plot(x, smooth1d(x, y, 50), 'b-', lw=4)
    xlim([0, 2 * pi])
    title("Smoothing with factor 50")



def test_smooth2d():
    """
        Simple test function for smooth2d function.

        Smoothing works pretty well, but there are still some visible lines in both x and y direction
        due to the 1D approach of the smooth2d function.
    """

    # test smoothing of function x*y with some random noise
    x = linspace(0, 1, 100)
    y = linspace(0, 1, 100)
    X, Y = meshgrid(x, y)
    F = X*Y

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            noise = 0.1 * (random.rand() - 0.5)
            F[i, j] += noise

    subplots(nrows=2, ncols=2)
    suptitle('Test of 2D smoothing. Function used: f = x*y, with random noise')

    subplot(2, 2, 1)
    pcolor(X, Y, X*Y)
    title('Original function f = x*y')

    subplot(2, 2, 2)
    pcolor(X, Y, F)
    title('Original function with random noise')

    subplot(2, 2, 3)
    pcolor(X, Y, smooth2d(X, Y, F, factor=2))
    title('Smoothed with factor 2')

    subplot(2, 2, 4)
    pcolor(X, Y, smooth2d(X, Y, F, factor=10))
    title('Smoothed with factor 10')



def main():
    run_tests()
    test_smooth1d()
    test_smooth2d()

    x = linspace(0., 2. * pi, 20)
    y = sin(x)
    # 1d example
    x = linspace(0., 2. * pi, 20)
    y = sin(x)
    spl1d = spline(x=x, f=y, dims=1)
    xx = linspace(0., 2. * pi, 100)
    figure()
    # function
    plot(xx, spl1d.eval1d(xx))
    plot(x, y, 'o', xx, sin(xx), 'r--')
    title('function')

    # 2d example
    fig = figure()
    ax = fig.add_subplot(121)
    x = linspace(0.0, 3.0, 11)
    y = linspace(0.0, 3.0, 11)
    X, Y = meshgrid(x, y)
    Z = (X + Y) * exp(-1.0 * (X * X + Y * Y))
    ax.pcolor(X, Y, Z)
    ax.set_title('original')

    spl2d = spline(x=x, y=y, f=Z, dims=2)
    # figure()
    ax2 = fig.add_subplot(122)
    x = linspace(0.0, 3.0, 51)
    y = linspace(0.0, 3.0, 51)
    X, Y = meshgrid(x, y)
    Z = spl2d.eval2d(x, y)
    ax2.pcolor(X, Y, Z)
    ax2.set_title('interpolated')

    # 3d example
    x = linspace(0.0, 3.0, 10)
    y = linspace(0.0, 3.0, 10)
    z = linspace(0.0, 3.0, 10)
    X, Y, Z = meshgrid(x, y, z)
    F = (X + Y + Z) * exp(-1.0 * (X * X + Y * Y + Z * Z))
    X, Y = meshgrid(x, y)
    fig3d = figure()
    ax = fig3d.add_subplot(121)
    ax.pcolor(X, Y, F[..., int(len(z) / 2)])
    ax.set_title('original')

    spl3d = spline(x=x, y=y, z=z, f=F, dims=3)
    x = linspace(0.0, 3.0, 50)
    y = linspace(0.0, 3.0, 50)
    z = linspace(0.0, 3.0, 50)
    X, Y = meshgrid(x, y)
    ax2 = fig3d.add_subplot(122)
    F = spl3d.eval3d(x, y, z)
    ax2.pcolor(X, Y, F[..., int(len(z) / 2)])
    ax2.set_title('interpolated')

    show()


# end main

if __name__ == "__main__":
    main()
