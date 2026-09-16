"""Benchmark of the two-way wave equation operator (``pytests/test_twoway.py``).

``AcousticWave2D`` wraps devito: a forward or adjoint pass is a full
finite-difference simulation and costs hundreds of milliseconds rather than the
few milliseconds of the other operators of the suite, hence the small model and
the longer timeout. The devito code generation and compilation happen in
``setup``, which warms both passes up, so they are not attributed to the
timings.
"""

import numpy as np

from .common import OperatorBenchmark, build, getop, require


class AcousticWave2D(OperatorBenchmark):
    """Born modelling in a constant-velocity model, 2 sources and 20 receivers."""

    timeout = 600
    nx, nz = 60, 50
    dx, dz = 10.0, 10.0
    ns, nr = 2, 20
    src_z = rec_z = 5.0
    tn = 500.0
    v0 = 2000.0

    def make_operator(self):
        require("devito")
        import devito

        devito.configuration["log-level"] = "ERROR"
        x = np.arange(self.nx) * self.dx
        sx = np.linspace(x.min(), x.max(), self.ns)
        rx = np.linspace(x.min(), x.max(), self.nr)
        return build(
            getop("waveeqprocessing.AcousticWave2D"),
            (self.nx, self.nz),
            (0.0, 0.0),
            (self.dx, self.dz),
            np.full((self.nx, self.nz), self.v0, dtype=np.float32),
            sx,
            self.src_z,
            rx,
            self.rec_z,
            0.0,
            self.tn,
            "Ricker",
            space_order=4,
            nbl=20,
            f0=15.0,
            dtype="float32",
        )
