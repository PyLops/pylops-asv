"""Benchmarks of ``pylops.waveeqprocessing`` operators
(``pytests/test_waveeqprocessing.py``, ``pytests/test_oneway.py``,
``pytests/test_wavedecomposition.py``, ``pytests/test_blending.py``)."""

import numpy as np

from .common import OperatorBenchmark, build, getop

DT = 0.004


class MDC(OperatorBenchmark):
    """Single-sided multi-dimensional convolution with ``nx`` virtual sources."""

    params = [["numpy", "scipy"]]
    param_names = ["fftengine"]
    par = {
        "ox": 0,
        "dx": 2,
        "nx": 100,
        "oy": 0,
        "dy": 2,
        "ny": 100,
        "ot": 0,
        "dt": DT,
        "nt": 512,
        "f0": 20,
    }

    def setup_cache(self):
        from pylops.utils.seismicevents import linear3d, makeaxis
        from pylops.utils.wavelets import ricker

        par = self.par
        t, _, x, y = makeaxis(par)
        wav = ricker(t[:41], f0=par["f0"])[0]
        t0 = np.array([25, 50, 75]) * par["dt"]
        _, Gwav = linear3d(
            x, y, t, 1500.0, t0, (0, 0, 0), (0, 0, 0), (1.0, 0.6, 2.0), wav
        )
        nfmax = int(np.ceil((par["nt"] + 1.0) / 2))
        Gwav_fft = np.fft.fft(Gwav, par["nt"], axis=-1)[..., :nfmax]
        return np.ascontiguousarray(Gwav_fft.transpose(2, 0, 1))

    def make_operator(self, G, fftengine):
        return build(
            getop("MDC"),
            G,
            nt=self.par["nt"],
            nv=self.par["nx"],
            dt=DT,
            dr=self.par["dx"],
            twosided=False,
            fftengine=fftengine,
        )


class PhaseShift(OperatorBenchmark):
    def make_operator(self):
        nt, nx = 256, 512
        freq = np.fft.rfftfreq(nt, DT)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, 10.0))
        return getop("PhaseShift")(1500.0, 200.0, nt, freq, kx)


class UpDownComposition2D(OperatorBenchmark):
    def make_operator(self):
        nt, nr = 256, 512
        return getop("UpDownComposition2D")(
            nt, nr, DT, 10.0, 1000.0, 1500.0, nffts=(nr, nt), critical=90.0, ntaper=5
        )


class UpDownComposition3D(OperatorBenchmark):
    def make_operator(self):
        nt, ny, nx = 128, 64, 64
        return getop("UpDownComposition3D")(
            nt,
            (ny, nx),
            DT,
            (10.0, 10.0),
            1000.0,
            1500.0,
            nffts=(ny, nx, nt),
            critical=90.0,
            ntaper=5,
        )


class PressureToVelocity(OperatorBenchmark):
    def make_operator(self):
        nt, nr = 256, 512
        return getop("PressureToVelocity")(
            nt,
            nr,
            DT,
            10.0,
            1000.0,
            1500.0,
            nffts=(nr, nt),
            critical=90.0,
            ntaper=5,
            topressure=False,
        )


class _Blending(OperatorBenchmark):
    nt, nr, ns = 501, 100, 100
    group_size = 2

    def times(self, continuous):
        rng = np.random.default_rng(0)
        if continuous:
            overlap = 0.5
            times = 2.0 * rng.random(self.ns) - 1.0
            times += np.arange(0, overlap * self.nt * self.ns, overlap * self.nt) * DT
            times[0] = 0.0
            return times
        return (0.8 * rng.random(self.ns)).reshape(self.group_size, -1)


class BlendingContinuous(_Blending):
    def make_operator(self):
        return getop("waveeqprocessing.BlendingContinuous")(
            self.nt, self.nr, self.ns, DT, self.times(continuous=True)
        )


class BlendingGroup(_Blending):
    def make_operator(self):
        return getop("waveeqprocessing.BlendingGroup")(
            self.nt,
            self.nr,
            self.ns,
            DT,
            self.times(continuous=False),
            n_groups=self.ns // self.group_size,
            group_size=self.group_size,
        )


class BlendingHalf(_Blending):
    def make_operator(self):
        return getop("waveeqprocessing.BlendingHalf")(
            self.nt,
            self.nr,
            self.ns,
            DT,
            self.times(continuous=False),
            n_groups=self.ns // self.group_size,
            group_size=self.group_size,
        )
