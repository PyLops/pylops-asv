"""Benchmarks of the combining operators (``pytests/test_combine.py``,
``pytests/test_kronecker.py``)."""

import numpy as np

from .common import OperatorBenchmark, getop


def _matrixmults(n, m, nops):
    rng = np.random.default_rng(0)
    MatrixMult = getop("MatrixMult")
    return [MatrixMult(rng.standard_normal((n, m))) for _ in range(nops)]


class VStack(OperatorBenchmark):
    def make_operator(self):
        return getop("VStack")(_matrixmults(1500, 1500, 2))


class HStack(OperatorBenchmark):
    def make_operator(self):
        return getop("HStack")(_matrixmults(1500, 1500, 2))


class BlockDiag(OperatorBenchmark):
    def make_operator(self):
        return getop("BlockDiag")(_matrixmults(1500, 1500, 2))


class Block(OperatorBenchmark):
    def make_operator(self):
        ops = _matrixmults(1000, 1000, 4)
        return getop("Block")([[ops[0], ops[1]], [ops[2], ops[3]]])


class Kronecker(OperatorBenchmark):
    def make_operator(self):
        ops = _matrixmults(300, 300, 2)
        return getop("Kronecker")(ops[0], ops[1])
