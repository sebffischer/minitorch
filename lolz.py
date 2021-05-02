import numpy as np


def f(arr):
    arr[0] = 100
    return None


arr = np.array([1, 2, 3])

f(arr)
print(arr)


def g(x):
    x += 1


x = 10
print(x)
g(x)
print(x)
