"""Benchmarks of stationary and non-stationary convolution operators
(``pytests/test_convolve.py``, ``pytests/test_nonstatconvolve.py``)."""

import numpy as np
from scipy.signal.windows import triang

from .common import OperatorBenchmark, getop, require

DIMS2D = (1000, 1000)


def _filt1d(n):
    return triang(n, sym=True)


def _filt2d(n):
    return np.outer(_filt1d(n), _filt1d(n))


def _filt3d(n):
    return np.einsum("i,j,k->ijk", _filt1d(n), _filt1d(n), _filt1d(n))


class Convolve1D(OperatorBenchmark):
    """1D convolution of a 1D signal (``direct`` and ``fft`` methods)."""

    params = [["direct", "fft"]]
    param_names = ["method"]

    def make_operator(self, method):
        return getop("Convolve1D")(1_000_000, h=_filt1d(51), offset=25, method=method)


class Convolve1DAlongAxis(OperatorBenchmark):
    """1D convolution applied along one axis of a 2D signal."""

    def make_operator(self):
        return getop("Convolve1D")(
            DIMS2D, h=_filt1d(51), offset=25, axis=-1, method="fft"
        )


class Convolve2D(OperatorBenchmark):
    params = [["direct", "fft"]]
    param_names = ["method"]

    def make_operator(self, method):
        return getop("Convolve2D")(
            (512, 512), h=_filt2d(9), offset=(4, 4), method=method
        )


class ConvolveND(OperatorBenchmark):
    def make_operator(self):
        return getop("ConvolveND")(
            (100, 100, 100), h=_filt3d(7), offset=(3, 3, 3), axes=(0, 1, 2)
        )


class NonStationaryConvolve1D(OperatorBenchmark):
    def make_operator(self):
        h = _filt1d(51)
        hs = np.vstack([h, -h, 2 * h, h, -h])
        return getop("NonStationaryConvolve1D")(
            dims=(200, 2000), hs=hs, ih=(200, 600, 1000, 1400, 1800), axis=-1
        )


class NonStationaryConvolve2D(OperatorBenchmark):
    params = [["numpy", "numba"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        h = _filt2d(7)
        hs = np.stack([h, -h, 2 * h] * 3).reshape(3, 3, 7, 7)
        return getop("NonStationaryConvolve2D")(
            dims=(128, 128),
            hs=hs,
            ihx=(32, 64, 96),
            ihz=(32, 64, 96),
            engine=engine,
        )


class NonStationaryConvolve3D(OperatorBenchmark):
    params = [["numpy", "numba"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        h = _filt3d(5)
        hs = np.stack([h, -h] * 4).reshape(2, 2, 2, 5, 5, 5)
        return getop("NonStationaryConvolve3D")(
            dims=(24, 24, 24),
            hs=hs,
            ihx=(6, 18),
            ihy=(6, 18),
            ihz=(6, 18),
            engine=engine,
        )


class NonStationaryFilters1D(OperatorBenchmark):
    def make_operator(self):
        x = np.random.default_rng(0).standard_normal(2000)
        return getop("NonStationaryFilters1D")(
            inp=x, hsize=51, ih=(400, 800, 1200, 1600)
        )


class NonStationaryFilters2D(OperatorBenchmark):
    def make_operator(self):
        x = np.random.default_rng(0).standard_normal((128, 128))
        return getop("NonStationaryFilters2D")(
            inp=x, hshape=(7, 7), ihx=(32, 64, 96), ihz=(32, 64, 96)
        )
