import minitorch
from strategies import assert_close


def neg(a):
    return -a


def mul(a, b):
    return a * b


f = neg


t = minitorch.tensor_fromlist([[2, 3], [4, 6], [5, 7]])
t2 = f(t)

for ind in t._tensor.indices():
    assert_close(t2[ind], f(minitorch.Scalar(t[ind])).data)


t1, t2 = t
t3 = mul(t1, t2)
for ind in t3._tensor.indices():
    assert t3[ind] == fn[1](minitorch.Scalar(t1[ind]), minitorch.Scalar(t2[ind])).data

