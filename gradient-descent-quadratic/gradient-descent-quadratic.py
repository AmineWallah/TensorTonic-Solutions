def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    def derivative(a, b, x):
        return 2 * a * x + b

    for step in range(steps):
        x0 = x0 - lr * derivative(a, b, x0)

    return x0
