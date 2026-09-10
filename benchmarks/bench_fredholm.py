"""Benchmarks of the Fredholm integral operator (``pytests/test_fredholm.py``)."""

import numpy as np

from .common import OperatorBenchmark, getop


class Fredholm1(OperatorBenchmark):
    params = [[True, False]]
    param_names = ["usematmul"]

    def setup_cache(self):
        return np.random.default_rng(0).standard_normal((50, 200, 200))

    def make_operator(self, G, usematmul):
        return getop("Fredholm1")(G, nz=100, saveGt=True, usematmul=usematmul)
