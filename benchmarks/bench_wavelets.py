"""Benchmarks of wavelet-like transforms (``pytests/test_dwts.py``,
``pytests/test_seislet.py``, ``pytests/test_pwd.py``, ``pytests/test_udct.py``).

``DTCWT`` is not benchmarked: dtcwt 0.14.0, its latest release, calls the
``np.asfarray`` function removed in numpy 2.
"""

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


class UDCT(OperatorBenchmark):
    """Uniform discrete curvelet transform (needs curvelets, pylops>=2.8.0).

    The real transform takes a real input and returns complex coefficients; the
    complex one separates positive and negative frequencies and is therefore
    roughly twice as large.
    """

    params = [["real", "complex"]]
    param_names = ["transform_kind"]

    def make_operator(self, transform_kind):
        require("curvelets")
        dtype = "float64" if transform_kind == "real" else "complex128"
        return getop("signalprocessing.UDCT")(
            dims=(512, 512), transform_kind=transform_kind, dtype=dtype
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
