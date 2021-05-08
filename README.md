# TODO

Check that the forward and backward functions are all properly implemented

## Some adjustments

Environment

- Slightly differrent environment to avoid numba segfault
- Updated pytest to latest version (07.05.21)

The backend is dynamically constructed because we then have to write the code only once
for CUDA and CPU ops (make_tensor-\_backend in tensor_functions.py)

Some comments

- pay attentino with the shape: .\_shape is an array and shape is a tuple

## conftest.py

In order to set default settings pytest allows the use of _conftest.py_.
In the original minitorch the profile was set in strategies.py but with confest.py
it is cleaner imo or conforms to standards.

## Debug failing hypothesis test cases

Debugging the failing hypothesis test cases can be done via the following:

1. Set print_blob=True (either with the @settings decorator or when defining the
   profile in conftest.py)
1. When this flag is set, the pytest output will give information on how to
   temporarily change your tests to reproduce the error. Then the

## Problem with zsh and debugging pytest

When debugging tests with zsh and using reparameterization there is an issue with zsh
A workaround for now is to use bash.
