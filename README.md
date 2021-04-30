# TODO

Check that the forward and backward functions are all properly implemented

# MiniTorch Module 1

<img src="https://minitorch.github.io/_images/match.png" width="100px">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module1.html

This module requires `operators.py` and `module.py` from Module 0

```
cp ../Module-0/operators.py ../Module-0/module.py minitorch/
```

- Tests:

```
python run_tests.py
```

How it works:
The Scalar Functions (e.g. Mul) inherit from ScalarFunction which inherits
from FunctionBase

FunctionBase has the apply function.

With the python data-model the basic mathematical operators are overwritten by
FunctionBase.apply(Mul) e.g.. In addition to executing the operation with respect
to the .data of the Scalar, returns a new variable containing the history.
The history contains the applied function, relevant context for the backward
pass as well as the input values (some overlap between context and
input).
When backpropagating we start with d_out = 1 and with the leaf-variable
We then look at the last applied function and calculate the derivatives with
respect to the input. THis will give us Inputs with derivatives which can
again be backpropagated further.
