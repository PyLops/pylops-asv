"""Benchmarks of ``pylops.avo`` modelling operators (``pytests/test_avo.py``,
``pytests/test_poststack.py``, ``pytests/test_prestack.py``)."""

import numpy as np
from scipy.signal import filtfilt

from .common import OperatorBenchmark, getop

DT0 = 0.004
NTWAV = 41


def _wavelet():
    from pylops.utils.wavelets import ricker

    t0 = np.arange(NTWAV) * DT0
    return ricker(t0[: NTWAV // 2 + 1], 20)[0]


def _elastic_model(nt0):
    """Log vp/vs/rho model (nt0, 3) as in ``pytests/test_prestack.py``."""
    rng = np.random.default_rng(0)
    smooth = np.ones(5) / 5.0
    vp = 1200 + np.arange(nt0) + filtfilt(smooth, 1, rng.normal(0, 40, nt0))
    vs = 600 + vp / 2 + filtfilt(smooth, 1, rng.normal(0, 20, nt0))
    rho = 1000 + vp + filtfilt(smooth, 1, rng.normal(0, 30, nt0))
    return np.stack((np.log(vp), np.log(vs), np.log(rho)), axis=1)


class AVOLinearModelling(OperatorBenchmark):
    params = [["akirich", "fatti"]]
    param_names = ["linearization"]

    def make_operator(self, linearization):
        theta = np.linspace(0, 40, 21)
        return getop("avo.AVOLinearModelling")(
            theta, vsvp=0.5, nt0=2000, spatdims=(50,), linearization=linearization
        )


class PoststackLinearModelling(OperatorBenchmark):
    params = [[False, True]]
    param_names = ["explicit"]

    def make_operator(self, explicit):
        return getop("PoststackLinearModelling")(
            _wavelet(), nt0=1000, spatdims=(500,), explicit=explicit
        )


class PrestackLinearModelling(OperatorBenchmark):
    params = [[False, True]]
    param_names = ["explicit"]

    def make_operator(self, explicit):
        theta = np.linspace(0, 40, 7)
        return getop("PrestackLinearModelling")(
            _wavelet(),
            theta,
            vsvp=0.5,
            nt0=500,
            spatdims=(200,),
            linearization="akirich",
            explicit=explicit,
            kind="centered",
        )


class PrestackWaveletModelling(OperatorBenchmark):
    def setup_cache(self):
        return _elastic_model(1000)

    def make_operator(self, m):
        theta = np.linspace(0, 40, 21)
        return getop("avo.PrestackWaveletModelling")(
            m, theta, nwav=NTWAV, wavc=NTWAV // 2, vsvp=0.5, linearization="akirich"
        )
