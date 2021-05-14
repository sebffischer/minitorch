from numba import prange, njit


@njit(parallel=True)
def f():
    for i in prange(10):
        for j in range(3):
            i = i * 10

        print(j ** 2)


f()


# @njit(inline="always")
# def g(x):
#    y = int(x)
#    for i in range(10):
#        x = x + 1
#
#    return x
#
#
# f()
#
#

x = 10

for i in range(10):
    x += 3
