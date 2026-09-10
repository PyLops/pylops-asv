"""Benchmarks of the derivative operators (``pytests/test_derivative.py``)."""

import numpy as np

from .common import OperatorBenchmark, getop

DIMS = (1000, 1000)
V = np.sqrt(2) / 2 * np.ones(2)


class FirstDerivative(OperatorBenchmark):
    params = [["forward", "centered", "backward"]]
    param_names = ["kind"]

    def make_operator(self, kind):
        return getop("FirstDerivative")(
            DIMS, axis=0, sampling=1.0, kind=kind, edge=True, order=3
        )


class SecondDerivative(OperatorBenchmark):
    def make_operator(self):
        return getop("SecondDerivative")(DIMS, axis=0, sampling=1.0, edge=True)


class Laplacian(OperatorBenchmark):
    def make_operator(self):
        return getop("Laplacian")(
            DIMS, axes=(0, 1), weights=(1, 1), sampling=(1, 1), edge=True
        )


class Gradient(OperatorBenchmark):
    def make_operator(self):
        return getop("Gradient")(DIMS, sampling=(1, 1), edge=True, kind="centered")


class FirstDirectionalDerivative(OperatorBenchmark):
    def make_operator(self):
        return getop("FirstDirectionalDerivative")(
            DIMS, v=V, sampling=(1, 1), edge=True, kind="centered"
        )


class SecondDirectionalDerivative(OperatorBenchmark):
    def make_operator(self):
        return getop("SecondDirectionalDerivative")(
            DIMS, v=V, sampling=(1, 1), edge=True
        )
