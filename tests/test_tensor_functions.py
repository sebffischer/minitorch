import minitorch
import pytest
from hypothesis import given, reproduce_failure, settings
from hypothesis.strategies import floats, lists
from .strategies import tensors, shaped_tensors, assert_close


def test_mul():
    t1 = minitorch.tensor_fromlist([[1.0, 2.0]])
    t2 = minitorch.tensor_fromlist([[-1.0], [3.0]])
    expected = minitorch.tensor_fromlist([[-1.0, -2.0], [3.0, 6.0]])
    print(t1.shape)
    print(t2.shape)
    observed = t1 * t2
    print(observed)
    print(expected)

    for ind in observed._tensor.indices():
        assert_close(expected[ind], observed[ind])
