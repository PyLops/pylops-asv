# pylops-asv

[![Benchmarks](https://github.com/PyLops/pylops-asv/actions/workflows/benchmarks.yaml/badge.svg)](https://github.com/PyLops/pylops-asv/actions/workflows/benchmarks.yaml)
[![Check](https://github.com/PyLops/pylops-asv/actions/workflows/check.yaml/badge.svg)](https://github.com/PyLops/pylops-asv/actions/workflows/check.yaml)

Continuous benchmarking of [PyLops](https://github.com/PyLops/pylops) operators with
[airspeed velocity](https://asv.readthedocs.io) (asv).

For every operator, the suite in `benchmarks/` measures the wall time and the peak memory
of the forward (`matvec`) and adjoint (`rmatvec`) passes. The results are published at

**https://pylops.github.io/pylops-asv/**

## How it works

- `benchmarks/` holds one `bench_*.py` module per family of operators. Every benchmark
  subclasses `benchmarks.common.OperatorBenchmark` and only implements `make_operator`;
  `time_forward`, `time_adjoint`, `peakmem_forward` and `peakmem_adjoint` are inherited.
- `asv.conf.json` points asv at the PyLops GitHub repository: for each benchmarked commit asv
  clones it, builds a wheel and installs it in a fresh `uv` environment with the latest
  release of every dependency in the matrix (numpy, scipy, numba, pyfftw, PyWavelets,
  scikit-fmm, curvelets, devito). A benchmark whose optional dependency is missing, or
  whose operator does not exist yet in the benchmarked release, is *skipped* rather
  than failed.
- The `PyLops-benchmarks` GitHub Action runs every night on the heads of the `master` and
  `dev` branches of PyLops and on the latest release tag (commits that already have results
  are skipped). Results are committed to `results/` on `main` and the website is rebuilt
  with `asv publish` and deployed to GitHub Pages.
- Older releases or arbitrary revision ranges can be benchmarked retroactively by triggering
  the same workflow manually (*Actions → PyLops-benchmarks → Run workflow*) with either the
  number of most recent release tags (`last_tags`) or a git revision range (`revisions`,
  e.g. `v2.7.0..master`). Tick `quick` for a smoke test that runs every benchmark once.

Benchmarks run on GitHub-hosted runners (`ubuntu-latest`), so timings are noisy across
runs: look at trends and at the peak-memory numbers rather than at single points.

## Adding a benchmark

When a new operator is added to PyLops, open a companion pull request here:

1. Add a class to the relevant `benchmarks/bench_*.py` module (or a new module) that
   subclasses `OperatorBenchmark` and implements `make_operator`, which returns the operator
   to benchmark. Choose the size of the operator such that a single forward pass takes
   between 1 and 100 milliseconds.
2. Resolve the operator with `getop("FFT")` (or `getop("signalprocessing.FFT")`) rather than
   importing it: operators that do not exist in an older PyLops release are then *skipped*
   instead of failing. Likewise use `build(cls, ...)` for keyword arguments introduced
   after a release and `require("numba")` for optional dependencies.
3. Use `params`/`param_names` to benchmark several engines or sizes, and `setup_cache` for
   expensive inputs shared by all parameters.
4. Validate the suite (nothing is executed) and lint it:

   ```bash
   make install
   make benchcheck
   make lint
   ```

## Running locally

Local runs benchmark a local checkout of PyLops (`PYLOPS_REPO`, default `../pylops`),
installed in editable mode in this project's environment. Results are written to the
git-ignored `.asv/results-local` directory and never uploaded: they are useful to develop
benchmarks or to compare two versions of an operator, but the numbers published by the
GitHub Action are the reference.

```bash
make bench BENCH_ARGS="--quick -b bench_ffts"     # only the FFT module, one iteration
make bench PYLOPS_REPO=~/src/pylops                # another checkout
make benchpreview                                  # browse http://127.0.0.1:8765/ (ASV_PORT=<port>)
```

`BENCH_ARGS` is passed verbatim to `asv run`; see `asv run --help` for the options
(`-b REGEX` to select benchmarks, `--quick`, `--profile`, ...).

## Repository layout

```
asv.conf.json        asv configuration (project repo, environment matrix, build commands)
benchmarks/          benchmark suite (bench_*.py) and shared base class (common.py)
ci/                  helpers used by the workflow and by `make bench`
results/             results committed by the PyLops-benchmarks workflow
.github/workflows/   benchmarks.yaml (nightly/manual runs + deploy), check.yaml (PRs)
```
