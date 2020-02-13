def first_derivative(function, x, dx):
    # calculates the first derivative based on equation (5) in [1]
    return (function(x + dx) - function(x - dx)) / (2 * dx)

def gradient(function, x, dx):
    grad = []
    for f in function:
        res.append(first_derivative(f, x, dx))
    return grad
