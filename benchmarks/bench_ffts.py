"""Benchmarks of Fourier-type operators (``pytests/test_ffts.py``,
``pytests/test_dct.py``, ``pytests/test_shift.py``)."""

from .common import OperatorBenchmark, getop, require


class FFT(OperatorBenchmark):
    params = [["numpy", "scipy", "fftw"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "fftw":
            require("pyfftw")
        return getop("FFT")(dims=(1000, 1024), axis=-1, real=True, engine=engine)


class FFT2D(OperatorBenchmark):
    params = [["numpy", "scipy"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        return getop("FFT2D")(dims=(1024, 1024), axes=(0, 1), real=True, engine=engine)


class FFTND(OperatorBenchmark):
    params = [["numpy", "scipy"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        return getop("FFTND")(
            dims=(128, 128, 128), axes=(0, 1, 2), real=True, engine=engine
        )


class DCT(OperatorBenchmark):
    def make_operator(self):
        return getop("DCT")(dims=(1000, 1000), type=2, axes=1)


class Shift(OperatorBenchmark):
    def make_operator(self):
        return getop("Shift")((1000, 1000), 3.5, axis=0, real=True)
