# numerical first order derivative, parameters are function, point and dx
def first_derivative(function, x, dx):
    # using definition of derivative
    return (function(x + dx) - function(x)) / dx
