import os
import pytest
from hypothesis import settings


settings.register_profile("minitorch", print_blob=True)
settings.load_profile("minitorch")

# Not sure what this is doing
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
