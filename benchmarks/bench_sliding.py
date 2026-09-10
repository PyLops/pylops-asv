"""Benchmarks of sliding-window and patching operators
(``pytests/test_sliding.py``, ``pytests/test_patching.py``).

The operator applied in each window is a ``Diagonal`` so that what is
measured is the windowing machinery (tapers, overlap-add) itself.
"""

import numpy as np

from .common import OperatorBenchmark, build, getop


def _window_op(n):
    return getop("Diagonal")(np.ones(n))


class Sliding1D(OperatorBenchmark):
    def make_operator(self):
        dimd, nwin, nover = 4096, 64, 32
        _, dim = build(
            getop("signalprocessing.sliding1d_design"),
            dimd,
            nwin,
            nover,
            nwin,
            verb=False,
        )[:2]
        return build(
            getop("Sliding1D"),
            _window_op(nwin),
            dim=dim,
            dimd=dimd,
            nwin=nwin,
            nover=nover,
            tapertype="hanning",
            savetaper=True,
        )


class Sliding2D(OperatorBenchmark):
    def make_operator(self):
        nt, npy, nwin, nover = 256, 1024, 64, 32
        _, dims = build(
            getop("signalprocessing.sliding2d_design"),
            (npy, nt),
            nwin,
            nover,
            (nwin, nt),
            verb=False,
        )[:2]
        return build(
            getop("Sliding2D"),
            _window_op(nwin * nt),
            dims=dims,
            dimsd=(npy, nt),
            nwin=nwin,
            nover=nover,
            tapertype="hanning",
            savetaper=True,
        )


class Sliding3D(OperatorBenchmark):
    def make_operator(self):
        nt, npy, npx, nwin, nover = 64, 256, 256, 32, 16
        _, dims = build(
            getop("signalprocessing.sliding3d_design"),
            (npy, npx, nt),
            (nwin, nwin),
            (nover, nover),
            (nwin, nwin, nt),
            verb=False,
        )[:2]
        return build(
            getop("Sliding3D"),
            _window_op(nwin * nwin * nt),
            dims=dims,
            dimsd=(npy, npx, nt),
            nwin=(nwin, nwin),
            nover=(nover, nover),
            nop=(nwin, nwin, nt),
            tapertype="hanning",
            savetaper=True,
        )


class Patch2D(OperatorBenchmark):
    def make_operator(self):
        npy, npt, nwin, nover = 1024, 1024, 64, 32
        _, dims = build(
            getop("signalprocessing.patch2d_design"),
            (npy, npt),
            (nwin, nwin),
            (nover, nover),
            (nwin, nwin),
            verb=False,
        )[:2]
        return build(
            getop("Patch2D"),
            _window_op(nwin * nwin),
            dims=dims,
            dimsd=(npy, npt),
            nwin=(nwin, nwin),
            nover=(nover, nover),
            nop=(nwin, nwin),
            tapertype="hanning",
            savetaper=True,
        )


class Patch3D(OperatorBenchmark):
    def make_operator(self):
        n, nwin, nover = 128, 32, 16
        _, dims = build(
            getop("signalprocessing.patch3d_design"),
            (n, n, n),
            (nwin,) * 3,
            (nover,) * 3,
            (nwin,) * 3,
            verb=False,
        )[:2]
        return build(
            getop("Patch3D"),
            _window_op(nwin**3),
            dims=dims,
            dimsd=(n, n, n),
            nwin=(nwin,) * 3,
            nover=(nover,) * 3,
            nop=(nwin,) * 3,
            tapertype="hanning",
            savetaper=True,
        )
