from minitorch.tensor_functions import TensorFunctions, zeros
from minitorch.tensor import Tensor
from minitorch.tensor_data import TensorData
from minitorch.tensor_ops import map
import numpy as np


def f(x):
    return x ** 2


fn = map(f)

v = np.array([1, 2, 3, 4], dtype=np.float32)

v_data = TensorData(storage=v, shape=(2, 2))
backend = TensorFunctions

a = Tensor(v=v_data, backend=backend)
print(a)
print(a.shape)

new = zeros((2, 2))
print(new)

# breakpoint()
x = fn(a)
print(x)
