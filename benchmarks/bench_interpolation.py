"""Benchmarks of interpolation and resampling operators
(``pytests/test_interpolation.py``, ``pytests/test_interpolation_spline.py``)."""

import numpy as np

from .common import OperatorBenchmark, getop

DIMS = (1000, 1000)


def _iava(n, nsub, fractional=True):
    rng = np.random.default_rng(0)
    iava = np.sort(rng.permutation(n - 1)[:nsub]).astype("float64")
    return iava + 0.3 if fractional else iava


class Interp(OperatorBenchmark):
    params = [["nearest", "linear", "sinc"]]
    param_names = ["kind"]

    def make_operator(self, kind):
        dims = (200, DIMS[1]) if kind == "sinc" else DIMS
        return getop("Interp")(dims, _iava(dims[0], dims[0] // 2), axis=0, kind=kind)[0]


class Bilinear(OperatorBenchmark):
    def make_operator(self):
        rng = np.random.default_rng(0)
        npts = 500_000
        iava = np.vstack(
            (rng.uniform(0, DIMS[0] - 1, npts), rng.uniform(0, DIMS[1] - 1, npts))
        )
        return getop("Bilinear")(iava, dims=DIMS)


class InterpCubicSpline(OperatorBenchmark):
    def make_operator(self):
        return getop("signalprocessing.InterpCubicSpline")(
            dims=DIMS, iava=_iava(DIMS[0], DIMS[0] // 2), axis=0
        )
