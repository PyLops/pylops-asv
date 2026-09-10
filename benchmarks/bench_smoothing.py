"""Benchmarks of smoothing and integration operators
(``pytests/test_smoothing.py``, ``pytests/test_causalintegration.py``)."""

from .common import OperatorBenchmark, getop

DIMS = (1000, 1000)


class Smoothing1D(OperatorBenchmark):
    def make_operator(self):
        return getop("Smoothing1D")(nsmooth=11, dims=DIMS, axis=0)


class Smoothing2D(OperatorBenchmark):
    def make_operator(self):
        return getop("Smoothing2D")(nsmooth=(7, 7), dims=DIMS, axes=(0, 1))


class SmoothingND(OperatorBenchmark):
    def make_operator(self):
        return getop("SmoothingND")(
            nsmooth=(5, 5, 5), dims=(100, 100, 100), axes=(0, 1, 2)
        )


class CausalIntegration(OperatorBenchmark):
    params = [["full", "half", "trapezoidal"]]
    param_names = ["kind"]

    def make_operator(self, kind):
        return getop("CausalIntegration")(
            DIMS, axis=-1, sampling=0.004, kind=kind, removefirst=False
        )
