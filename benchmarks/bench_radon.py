"""Benchmarks of Radon-type operators (``pytests/test_radon.py``,
``pytests/test_fourierradon.py``, ``pytests/test_chirpradon.py``)."""

import numpy as np

from .common import OperatorBenchmark, getop, require

DT, DH = 0.004, 10.0
PXMAX, PYMAX = 1e-3, 1e-3


def _axes2d(nt=501, nh=101, npx=51):
    t = np.arange(nt) * DT
    h = np.arange(nh) * DH
    px = np.linspace(0, PXMAX, npx)
    return t, h, px


def _axes3d(nt=201, nh=21, npx=11):
    t = np.arange(nt) * DT
    hy = np.arange(nh) * DH
    hx = np.arange(nh) * DH
    py = np.linspace(0, PYMAX, npx)
    px = np.linspace(0, PXMAX, npx)
    return t, hy, hx, py, px


class Radon2D(OperatorBenchmark):
    params = [["numpy", "numba"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        t, h, px = _axes2d()
        return getop("Radon2D")(
            t,
            h,
            px,
            centeredh=True,
            interp=True,
            kind="linear",
            onthefly=False,
            engine=engine,
        )


class Radon3D(OperatorBenchmark):
    params = [["numpy", "numba"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        t, hy, hx, py, px = _axes3d()
        return getop("Radon3D")(
            t,
            hy,
            hx,
            py,
            px,
            centeredh=True,
            interp=True,
            kind="linear",
            onthefly=False,
            engine=engine,
        )


class FourierRadon2D(OperatorBenchmark):
    params = [["numpy", "numba"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        t, h, px = _axes2d()
        nfft = int(2 ** np.ceil(np.log2(t.size)))
        return getop("signalprocessing.FourierRadon2D")(
            t, h, px, nfft, kind="linear", flims=[0, t.size // 2], engine=engine
        )


class FourierRadon3D(OperatorBenchmark):
    params = [["numpy", "numba"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        t, hy, hx, py, px = _axes3d()
        nfft = int(2 ** np.ceil(np.log2(t.size)))
        return getop("signalprocessing.FourierRadon3D")(
            t,
            hy,
            hx,
            py,
            px,
            nfft,
            kind=("linear", "linear"),
            flims=[0, t.size // 2],
            engine=engine,
        )


class ChirpRadon2D(OperatorBenchmark):
    def make_operator(self):
        t, h, _ = _axes2d(nt=1001, nh=201)
        return getop("ChirpRadon2D")(t, h, 2e-2)


class ChirpRadon3D(OperatorBenchmark):
    params = [["numpy", "fftw"]]
    param_names = ["engine"]

    def make_operator(self, engine):
        kwargs = {}
        if engine == "fftw":
            require("pyfftw")
            kwargs = {"flags": ("FFTW_ESTIMATE",), "threads": 1}
        t, hy, hx, _, _ = _axes3d(nt=201, nh=41)
        return getop("ChirpRadon3D")(t, hy, hx, (1e-2, 2e-2), engine=engine, **kwargs)
