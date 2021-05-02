"""
Test some of the backward operators. Not sure whether they are well implemented
for small values. Note that the mark must be added to pytest.ini
"""
from hypothesis import given
import pytest
from .strategies import scalars, assert_close, small_floats, floats
import minitorch.operators as operators
from minitorch.scalar import central_difference


@pytest.mark.backward
@given(floats(min_value=-100, max_value=100))
def test_inv(x):
    if abs(x) > 0.01:
        x0 = central_difference(operators.inv, x)
        x1 = operators.inv_back(x, 1)
        assert_close(x0, x1)


@pytest.mark.backward
@given(floats(min_value=0.1, max_value=100))
def test_log(x):
    x0 = central_difference(operators.log, x)
    x1 = operators.log_back(x, 1)
    assert_close(x0, x1)


@pytest.mark.backward
@given(floats(min_value=0.1, max_value=100))
def test_exp(x):
    x0 = central_difference(operators.exp, x)
    x1 = operators.exp_back(x, 1)
    assert_close(x0, x1)


@pytest.mark.backward
@given(floats(min_value=-100, max_value=100))
def test_max(x):
    x0 = central_difference(operators.max, x)
    x1 = operators.test_max(x, 1)
    assert_close(x0, x1)

