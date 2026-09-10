"""Benchmarks of ``pylops.basicoperators`` (see ``pytests/test_basicoperators.py``
and friends)."""

import numpy as np

from .common import OperatorBenchmark, getop, require

N1D = 1_000_000
DIMS2D = (1000, 1000)


class FunctionOperator(OperatorBenchmark):
    params = [["float64", "complex128"]]
    param_names = ["dtype"]

    def make_operator(self, dtype):
        rng = np.random.default_rng(0)
        G = rng.standard_normal((2000, 1000)).astype(dtype)
        if dtype == "complex128":
            G = G + 1j * rng.standard_normal((2000, 1000))
        return getop("FunctionOperator")(
            lambda x: G @ x, lambda y: G.T.conj() @ y, 2000, 1000, dtype=dtype
        )


class MemoizeOperator(OperatorBenchmark):
    """Repeated evaluation of the same vector, i.e. the cached path."""

    def make_operator(self):
        G = np.random.default_rng(0).standard_normal((2000, 2000))
        return getop("MemoizeOperator")(getop("MatrixMult")(G), max_neval=2)


class Regression(OperatorBenchmark):
    def make_operator(self):
        t = np.arange(N1D, dtype="float64") / N1D
        return getop("Regression")(t, order=4)


class LinearRegression(OperatorBenchmark):
    def make_operator(self):
        return getop("LinearRegression")(np.arange(N1D, dtype="float64") / N1D)


class MatrixMult(OperatorBenchmark):
    params = [["float64", "complex128"]]
    param_names = ["dtype"]

    def make_operator(self, dtype):
        rng = np.random.default_rng(0)
        G = rng.standard_normal((4000, 2000))
        if dtype == "complex128":
            G = G + 1j * rng.standard_normal((4000, 2000))
        return getop("MatrixMult")(G.astype(dtype), dtype=dtype)


class MatrixMultOtherdims(OperatorBenchmark):
    def make_operator(self):
        G = np.random.default_rng(0).standard_normal((2000, 1000))
        return getop("MatrixMult")(G, otherdims=8)


class Diagonal(OperatorBenchmark):
    def make_operator(self):
        d = np.random.default_rng(0).standard_normal(N1D)
        return getop("Diagonal")(d)


class DiagonalBroadcast(OperatorBenchmark):
    def make_operator(self):
        d = np.random.default_rng(0).standard_normal(DIMS2D[0])
        return getop("Diagonal")(d, dims=DIMS2D, axis=0)


class Zero(OperatorBenchmark):
    def make_operator(self):
        return getop("Zero")(N1D, N1D)


class Identity(OperatorBenchmark):
    params = [[True, False]]
    param_names = ["inplace"]

    def make_operator(self, inplace):
        return getop("Identity")(N1D, N1D, inplace=inplace)


class Restriction(OperatorBenchmark):
    def make_operator(self):
        rng = np.random.default_rng(0)
        iava = np.sort(rng.permutation(DIMS2D[0])[: DIMS2D[0] // 2])
        return getop("Restriction")((DIMS2D[0], 2 * DIMS2D[1]), iava, axis=0)


class Flip(OperatorBenchmark):
    def make_operator(self):
        return getop("Flip")(DIMS2D, axis=0)


class Symmetrize(OperatorBenchmark):
    def make_operator(self):
        return getop("Symmetrize")(DIMS2D, axis=0)


class Roll(OperatorBenchmark):
    def make_operator(self):
        return getop("Roll")(DIMS2D, axis=0, shift=17)


class Transpose(OperatorBenchmark):
    def make_operator(self):
        return getop("Transpose")(dims=(100, 100, 100), axes=(2, 1, 0))


class Pad(OperatorBenchmark):
    def make_operator(self):
        return getop("Pad")(dims=DIMS2D, pad=((10, 10), (20, 20)))


class Sum(OperatorBenchmark):
    def make_operator(self):
        return getop("Sum")(dims=DIMS2D, axis=0)


class Real(OperatorBenchmark):
    def make_operator(self):
        return getop("Real")(dims=DIMS2D, dtype="complex128")


class Imag(OperatorBenchmark):
    def make_operator(self):
        return getop("Imag")(dims=DIMS2D, dtype="complex128")


class Conj(OperatorBenchmark):
    def make_operator(self):
        return getop("Conj")(dims=DIMS2D, dtype="complex128")


class Spread(OperatorBenchmark):
    """Spread along linear events (same look-up table as ``Radon2D``)."""

    params = [["numpy", "numba"]]
    param_names = ["engine"]
    nt, nh, npx = 501, 101, 51

    def setup_cache(self):
        px = np.linspace(0, 0.5, self.npx)
        it = np.arange(self.nt)[:, np.newaxis]
        ih = np.arange(self.nh)[np.newaxis, :]
        table = np.full((self.npx, self.nt, self.nh), np.nan, dtype=np.float32)
        for ipx, p in enumerate(px):
            tt = np.floor(it + p * ih)
            valid = tt < self.nt
            table[ipx][valid] = tt[valid]
        return table

    def make_operator(self, table, engine):
        if engine == "numba":
            require("numba")
        return getop("Spread")(
            dims=(self.npx, self.nt),
            dimsd=(self.nh, self.nt),
            table=table,
            engine=engine,
        )
