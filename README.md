# pylops-asv

[![Benchmarks](https://github.com/PyLops/pylops-asv/actions/workflows/benchmarks.yaml/badge.svg)](https://github.com/PyLops/pylops-asv/actions/workflows/benchmarks.yaml)
[![Check](https://github.com/PyLops/pylops-asv/actions/workflows/check.yaml/badge.svg)](https://github.com/PyLops/pylops-asv/actions/workflows/check.yaml)
[![Regressions](https://github.com/PyLops/pylops-asv/actions/workflows/regressions.yaml/badge.svg)](https://github.com/PyLops/pylops-asv/actions/workflows/regressions.yaml)

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
  the same workflow manually (*Actions → PyLops-benchmarks → Run workflow*) with the number
  of most recent release tags (`last_tags`), an explicit list of revisions (`commits`, e.g.
  `v2.8.0 dev~3`, which is how the whole history is re-measured when the way of measuring
  changes) or a git revision range (`revisions`, e.g. `v2.7.0..master`). Tick `quick` for a
  smoke test that runs every benchmark once.
- Two independent switches control what is skipped. `skip_commits` drops the commits that
  already have a results file, and `skip_benchmarks` drops, within each selected commit,
  the benchmarks that already have a successful result. **To back-fill a benchmark that
  has just been added to the suite, untick `skip_commits` and leave `skip_benchmarks`
  ticked**: the commits are revisited but only the missing benchmarks run. Unticking both
  re-benchmarks everything from scratch.

Benchmarks run on GitHub-hosted runners (`ubuntu-latest`), so timings are noisier than on
a dedicated machine: look at trends and at the peak-memory numbers rather than at single
points. Two measures keep the hardware out of the numbers as much as possible:

- **Every numerical library is restricted to one thread** (`OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
  `NUMBA_NUM_THREADS`) and the benchmarks are **pinned to a single core** with `taskset`.
  Runners differ in how many cores they offer, so a multi-threaded kernel measures the
  machine rather than the operator; peak performance is not what is tracked here,
  reproducibility is. `make bench` pins the same variables, so local numbers can be read
  against the published ones. Results measured before this was introduced are not
  comparable with the ones after it, which is why the whole history was re-benchmarked
  rather than left as a series with a step in the middle.
- **The hardware of each run is recorded** by `ci/record_runner.py` in
  `runners/<commit>.json`: CPU model, flags, caches, core count, RAM and the thread
  settings in force. It deliberately does not go into asv's `machine.json`, as asv draws
  one line per distinct set of machine parameters and the published series would then
  split into one line per runner model. The file is overwritten when a commit is
  re-benchmarked, so it always describes the runner behind the results currently
  committed.

## Regression report

Each commit is benchmarked in its own job, hence on a different runner: comparing two
commits directly measures the hardware as well as the code. Comparing the heads of
`master` or `dev` with the previous night at face value used to flag a hundred
benchmarks, most of which recovered the next night. Single-threaded, core-pinned runs cut
most of that noise, and `runners/` tells which CPU measured what, but two runners of a
different model still differ: the report therefore keeps normalising.

The `PyLops-regressions` workflow runs every Monday (and on demand, *Actions →
PyLops-regressions → Run workflow*) and writes a report to its
[run summary](../../actions/workflows/regressions.yaml). Nothing is benchmarked: it only
reads `results/`, and `ci/regression_report.py` keeps the hardware out of the picture by

- **normalising** each commit by the median shift of comparable benchmarks (grouped by
  order of magnitude, since runners differ far more on the sub-millisecond benchmarks
  dominated by Python overhead than on the numerically heavy ones), and
- only *confirming* a regression once it **persists** over at least two commits, i.e. was
  reproduced on two runners. A step at the newest commit alone is listed separately, to be
  settled by the next nightly run.

A confirmed regression is reported as a warning annotation on the run, which stays green.
The report also lists the benchmarks that stopped producing a result and, for reference,
how much the runners differed. The same report can be obtained locally with

```bash
python3 ci/regression_report.py --repo ../pylops        # --repo only names the commits
python3 ci/regression_report.py --threshold 1.3         # flag smaller slowdowns
```

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
ci/                  helpers used by the workflows and by `make bench`
results/             results committed by the PyLops-benchmarks workflow
runners/             hardware that measured each commit (one json per commit)
.github/workflows/   benchmarks.yaml (nightly/manual runs + deploy), regressions.yaml
                     (weekly regression report), check.yaml (PRs)
```
