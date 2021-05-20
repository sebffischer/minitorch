from numba import prange, njit


# @njit(parallel=True)
# def f():
#    for i in prange(10):
#        for j in range(3):
#            i = i * 10
#
#        print(j ** 2)
#
#
# f()


@njit(parallel=True)
def f():
    a = 0
    for i in prange(2):
        x = int(i)
        for i in range(3): 
            x = 
            
        a = a + g(i)

    return a


@njit(inline="always")
def g(x):
    # x = int(x)
    for i in range(3):
        x = x + 1
    return x


f()
