import minitorch
from strategies import assert_close


# def neg(a):
#    return -a
#
#
# def mul(a, b):
#    return a * b
#
#
# f = neg
#
#
# t = minitorch.tensor_fromlist([[2, 3], [4, 6], [5, 7]])
# t2 = f(t)
#
# for ind in t._tensor.indices():
#    assert_close(t2[ind], f(minitorch.Scalar(t[ind])).data)
#
#
# t1, t2 = t, t
# t3 = mul(t1, t2)
# for ind in t3._tensor.indices():
#    assert t3[ind] == mul[1](minitorch.Scalar(t1[ind]), minitorch.Scalar(t2[ind])).data


# shape (3, 2)
t = minitorch.tensor_fromlist([[2, 3], [4, 6], [5, 7]])
print(t)

# here 1 means reduce the 1st dim, 2 -> 1
t_summed_2 = t.sum(1)
print(t_summed_2)

# shape (3, 1)
t_sum_2_expected = minitorch.tensor_fromlist([[5], [10], [12]])

# for ind in t_summed_2._tensor.indices():
#    assert_close(t_summed_2[ind], t_sum_2_expected[ind])
