# TODO

Check that the forward and backward functions are all properly implemented

## Some adjustments

- Slightly differrent environment to avoid numba segfault

- Run an individual test with

```python
# correct
python -m pytest tests/test.py
# wrong (did not use the right env)
pytest tests/test.py

```

- Note that there are also some difficulties when debugging the tests in combination
  with relative imports and the VSCode debugger.
  Because of that I changed from .strategies to from strategies
  --> This also means that tests must be run from the ../tests directory
  --> Find proper fix later

```python
# cd into tests
python -m pytest test.py -k taskX_Y
```

The backend is dynamically constructed because we then have to write the code only once
for CUDA and CPU ops (make_tensor-\_backend in tensor_functions.py)
