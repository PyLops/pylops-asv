"""Benchmarks of the Kirchhoff demigration operator
(``pytests/test_kirchhoff.py``)."""

import numpy as np

from .common import OperatorBenchmark, build, getop, require

V0 = 1500.0


def _geometry(nz, nx, nt, nsx, nrx, dz=5.0, dx=5.0, dt=0.002):
    z = np.arange(nz) * dz
    x = np.arange(nx) * dx
    t = np.arange(nt) * dt
    sx = np.linspace(x.min(), x.max(), nsx)
    rx = np.linspace(x.min(), x.max(), nrx)
    s2d = np.vstack((sx, 2 * np.ones(nsx)))
    r2d = np.vstack((rx, 2 * np.ones(nrx)))
    return z, x, t, s2d, r2d


class Kirchhoff(OperatorBenchmark):
    """2D Kirchhoff with analytical traveltimes in a constant velocity model."""

    params = [["numpy", "numba"]]
    param_names = ["engine"]
    timeout = 600

    def make_operator(self, engine):
        if engine == "numba":
            require("numba")
        from pylops.utils.wavelets import ricker

        z, x, t, s2d, r2d = _geometry(nz=40, nx=60, nt=200, nsx=6, nrx=12)
        wav, _, wavc = ricker(t[:41], f0=20)
        return build(
            getop("waveeqprocessing.Kirchhoff"),
            z,
            x,
            t,
            s2d,
            r2d,
            V0,
            wav,
            wavc,
            y=None,
            mode="analytic",
            engine=engine,
        )


class KirchhoffEikonal(OperatorBenchmark):
    """2D Kirchhoff with eikonal traveltimes (needs scikit-fmm)."""

    timeout = 600

    def make_operator(self):
        require("skfmm")
        from pylops.utils.wavelets import ricker

        nz, nx = 40, 60
        z, x, t, s2d, r2d = _geometry(nz=nz, nx=nx, nt=200, nsx=6, nrx=12)
        wav, _, wavc = ricker(t[:41], f0=20)
        vel = V0 * np.ones((nx, nz))
        return build(
            getop("waveeqprocessing.Kirchhoff"),
            z,
            x,
            t,
            s2d,
            r2d,
            vel,
            wav,
            wavc,
            y=None,
            mode="eikonal",
            engine="numpy",
        )
