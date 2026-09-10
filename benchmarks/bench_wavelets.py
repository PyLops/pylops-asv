"""Benchmarks of wavelet-like transforms (``pytests/test_dwts.py``,
``pytests/test_seislet.py``, ``pytests/test_pwd.py``)."""

import numpy as np

from .common import OperatorBenchmark, getop, require


class DWT(OperatorBenchmark):
    params = [["haar", "db4"]]
    param_names = ["wavelet"]

    def make_operator(self, wavelet):
        require("pywt")
        return getop("DWT")(dims=(1024, 1024), axis=0, wavelet=wavelet, level=3)


class DWT2D(OperatorBenchmark):
    def make_operator(self):
        require("pywt")
        return getop("DWT2D")(dims=(1024, 1024), axes=(0, 1), wavelet="haar", level=3)


class DWTND(OperatorBenchmark):
    def make_operator(self):
        require("pywt")
        return getop("signalprocessing.DWTND")(
            dims=(128, 128, 128), axes=(0, 1, 2), wavelet="haar", level=3
        )


class Seislet(OperatorBenchmark):
    params = [["haar", "linear"]]
    param_names = ["kind"]

    def setup_cache(self):
        return np.random.default_rng(0).normal(0, 0.1, (16, 128))

    def make_operator(self, slopes, kind):
        return getop("Seislet")(slopes, sampling=(10.0, 0.004), level=None, kind=kind)


class PWSprayer2D(OperatorBenchmark):
    def make_operator(self):
        return getop("signalprocessing.PWSprayer2D")(
            dims=(512, 1024), sigma=np.zeros((512, 1024))
        )


class PWSmoother2D(OperatorBenchmark):
    def make_operator(self):
        return getop("signalprocessing.PWSmoother2D")(
            dims=(512, 1024), sigma=np.zeros((512, 1024))
        )
