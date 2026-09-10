"""Benchmarks of ``pylops.medical`` operators (``pytests/test_mri.py``).

``CT2D`` requires astra-toolbox, which is not part of the benchmark
environment, and is therefore not benchmarked.
"""

import numpy as np

from .common import OperatorBenchmark, getop


class MRI2D(OperatorBenchmark):
    params = [["numpy", "scipy"]]
    param_names = ["fft_engine"]
    dims = (512, 512)

    def setup_cache(self):
        rng = np.random.default_rng(0)
        mask = np.zeros(self.dims, dtype=bool)
        nselected = int(0.3 * mask.size)
        mask.flat[rng.choice(mask.size, nselected, replace=False)] = True
        return mask.astype("complex128")

    def make_operator(self, mask, fft_engine):
        return getop("medical.MRI2D")(
            dims=self.dims, mask=mask, fft_engine=fft_engine, dtype="complex128"
        )
