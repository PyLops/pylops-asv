"""Shared machinery for the PyLops asv benchmarks.

Every operator benchmark subclasses :class:`OperatorBenchmark` and implements
``make_operator``; the base class then provides the four asv benchmarks
(``time_forward``, ``time_adjoint``, ``peakmem_forward``, ``peakmem_adjoint``)
for free. The base class is abstract so that asv does not collect it.

Skipping
--------
asv marks a benchmark as *skipped* when ``setup`` raises
``NotImplementedError``. This is used for:

* optional dependencies not installed in the benchmark environment
  (:func:`require`),
* operators that do not exist in the (possibly older) pylops release being
  benchmarked (:func:`getop`),
* keyword arguments that were added to an operator after the release being
  benchmarked (:func:`build`).

Peak memory
-----------
``peakmem_*`` benchmarks report the peak resident set size of the whole
benchmark process, hence they include the memory needed to build the operator
in ``setup`` (e.g. the dense matrix of ``MatrixMult`` or the look-up table of
``Spread``). This is deterministic for a given commit, so regressions are
still visible. Expensive inputs are computed in ``setup_cache`` whenever
possible so that they are not attributed to a single benchmark.
"""

import abc

import numpy as np
import pylops

try:
    from pylops.utils import deps
except ImportError:  # very old releases
    deps = None

_FLAGS = {
    "numba": "numba_enabled",
    "pyfftw": "pyfftw_enabled",
    "pywt": "pywt_enabled",
    "skfmm": "skfmm_enabled",
}


def require(*flags: str) -> None:
    """Skip the benchmark unless all optional dependencies are available."""
    for flag in flags:
        if deps is None or not getattr(deps, _FLAGS[flag], False):
            msg = f"{flag} is not available"
            raise NotImplementedError(msg)


_SUBPACKAGES = (
    "basicoperators",
    "signalprocessing",
    "waveeqprocessing",
    "avo",
    "medical",
)


def getop(path: str):
    """Return ``pylops.<path>`` or skip if it does not exist in this release.

    Parameters
    ----------
    path : str
        Dotted path relative to the ``pylops`` package, e.g.
        ``"signalprocessing.FourierRadon2D"`` or ``"medical.MRI2D"``. A bare
        name (e.g. ``"FFT"``) is looked up in ``pylops`` and then in every
        operator subpackage.
    """
    candidates = (
        [path] if "." in path else [path] + [f"{sub}.{path}" for sub in _SUBPACKAGES]
    )
    for candidate in candidates:
        obj = pylops
        for part in candidate.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    msg = f"pylops.{path} is not available in pylops {pylops.__version__}"
    raise NotImplementedError(msg)


def build(cls, *args, **kwargs):
    """Instantiate ``cls``; a ``TypeError`` (unknown keyword in an older
    release) skips the benchmark instead of failing it."""
    try:
        return cls(*args, **kwargs)
    except TypeError as e:
        msg = f"{getattr(cls, '__name__', cls)} cannot be built: {e}"
        raise NotImplementedError(msg) from e


def random_vector(rng, n: int, dtype) -> np.ndarray:
    """Random vector of size ``n`` with the given dtype (complex if needed)."""
    dtype = np.dtype(dtype)
    v = rng.standard_normal(n)
    if np.issubdtype(dtype, np.complexfloating):
        v = v + 1j * rng.standard_normal(n)
    return v.astype(dtype)


class OperatorBenchmark(abc.ABC):
    """Forward/adjoint time and peak-memory benchmarks of a linear operator.

    Subclasses implement :meth:`make_operator`, which receives the same
    positional arguments asv passes to ``setup`` (the ``setup_cache`` result
    first, if defined, followed by the values of ``params``).
    """

    # asv attributes shared by all benchmarks: keep the suite bounded
    timeout = 300
    rounds = 1
    repeat = (2, 5, 2.0)
    number = 0
    sample_time = 0.05
    warmup_time = 0.1
    min_run_count = 2

    @abc.abstractmethod
    def make_operator(self, *args):
        """Return the ``LinearOperator`` to benchmark."""

    def setup(self, *args):
        self.op = self.make_operator(*args)
        rng = np.random.default_rng(0)
        self.x = random_vector(rng, self.op.shape[1], self.op.dtype)
        self.y = random_vector(rng, self.op.shape[0], self.op.dtype)
        # warm up outside of the timed region (numba JIT, FFTW plans, caches)
        self.op.matvec(self.x)
        self.op.rmatvec(self.y)

    def time_forward(self, *args):
        self.op.matvec(self.x)

    def time_adjoint(self, *args):
        self.op.rmatvec(self.y)

    def peakmem_forward(self, *args):
        self.op.matvec(self.x)

    def peakmem_adjoint(self, *args):
        self.op.rmatvec(self.y)
