import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if sum(p) != 1: raise ValueError
    result = 0
    for i in range(len(x)):
        result += x[i] * p[i]

    return result
