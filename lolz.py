import numpy as np
from minitorch.operators import prod


def count(position, shape, out_index):
    """
    Convert a `position` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        position (int): current position.
        shape (tuple): tensor shape.
        out_index (array): the index corresponding to position.

    Returns:
      None : Fills in `out_index`.
    """
    # Assume we have shape (3, 4, 2) and position 7.
    # The mapping is unique if we simply make it increasing in d1, d2, d3...
    # Moving once completely along the first dimension would corrspond to 8 positions
    # --> (7 + 1) %
    # Basically to get the first index we need to get how often 4 * 2
    # fits into (position + 1). Then in order to get the second position we need
    # to figure out how often 2 fits into the remainder of the previous operation
    # Then the remainder givers us the last index
    # So
    for i in range(len(shape)):
        step_size = prod(shape[(i + 1) :])
        steps = position // step_size
        position -= steps * step_size
        out_index[i] = steps

    return None


arr = np.array([0, 0, 0])
position = 10
shape = (3, 2, 4)
num_el = int(prod(shape))


for i in range(num_el):
    count(i, shape, arr)
    print(arr)
