"""Report worrying performance regressions in the committed asv results.

Used by ``.github/workflows/regressions.yaml``, which writes the markdown
report to the run summary. Nothing is benchmarked here: the script only reads
``results/<machine>/*.json`` and compares the commits already measured, oldest
to newest (by pylops commit date).

Every commit is benchmarked in its own matrix job, hence on a different
GitHub-hosted runner, and ``machine.json`` does not record the CPU. Raw
commit-to-commit ratios are therefore dominated by the hardware: a plain
"1.5x slower than the previous commit" filter flags well over a hundred
benchmarks, most of which recover at the next commit. Two ideas keep the noise
out of the report:

* **Normalisation.** The hardware shifts whole groups of benchmarks at once,
  and by an amount that depends on the regime (runners differ much more on
  sub-millisecond benchmarks, dominated by Python overhead, than on the
  numerically heavy ones). Benchmarks are therefore bucketed by magnitude, and
  the median shift of a bucket at a given commit -- the *runner factor* -- is
  subtracted from every benchmark in it. What is left is specific to the
  benchmark, so a real regression stands out from the hardware.
* **Persistence.** A regression is only *confirmed* when it survives into at
  least two commits, i.e. it was reproduced on at least two runners; a step at
  the newest commit alone is reported separately, as it cannot be told apart
  from an unlucky run until the next nightly run.

Peak-memory results carry no runner factor (allocations do not depend on the
CPU), so they are compared directly against a tighter threshold.

Better than normalising, when it is available: every job also benchmarks a fixed
*anchor* commit on its own runner, and ``anchors/<commit>.json`` holds those
reference timings. The anchor code never changes, so whatever moves its timings
is the machine, and subtracting them cancels the runner exactly, per benchmark,
rather than estimating it from benchmarks that may have changed themselves. The
report falls back to the estimate for the commits and benchmarks an anchor does
not cover.

Normalisation cannot rescue every comparison. ``runners/<commit>.json`` records
what measured each commit, and the hardware turns out to differ in kind and not
only in speed: two runners reporting the very same CPU model can expose or mask
AVX-512, which is worth a factor of three to six on the numerically heavy
benchmarks. A step whose commits before and after it were measured on different
hardware is therefore reported apart, as a comparison the hardware explains just
as well as the code would.
"""

import argparse
import itertools
import json
import math
import pathlib
import statistics
import subprocess
import sys

HASH_LENGTH = 8  # must match "hash_length" in asv.conf.json
# Boundaries (in seconds) between the timing regimes used to bucket benchmarks
TIME_BUCKETS = (1e-4, 1e-3, 1e-2, 1e-1)
# Below this many benchmarks a bucket's runner factor is not robust, and the
# factor of the whole unit is used instead
MIN_BUCKET = 5
# Commits needed before a step to call the level in front of it a baseline
MIN_BASELINE = 2
# Number of most recent commits shown in the timeline of a flagged benchmark
TIMELINE = 6


class Result:
    """One benchmarked commit: its identity and its measurements.

    ``values`` maps ``(benchmark, params)`` to the measured value, in seconds
    for ``time_*`` benchmarks and in bytes for ``peakmem_*`` ones. Benchmarks
    that errored or were skipped (asv stores ``null`` for both) are absent.
    """

    def __init__(self, commit: str, date: int) -> None:
        self.commit = commit
        self.date = date
        self.label = commit
        self.envs: set[str] = set()
        self.values: dict[tuple[str, str], float] = {}

    def __repr__(self) -> str:
        return f"Result({self.label}, {len(self.values)} values)"


def read_results(results_dir: pathlib.Path, machine: str) -> list[Result]:
    """Load every results file of ``machine``, one ``Result`` per commit.

    A commit benchmarked in several environments (e.g. before and after a
    dependency was added to the matrix) has one file per environment; they are
    merged, the file with the most measurements winning the keys it shares with
    the others.
    """
    machine_dir = results_dir / machine
    if not machine_dir.is_dir():
        msg = f"no results directory for machine {machine!r} in {results_dir}"
        raise SystemExit(msg)

    byhash: dict[str, Result] = {}
    files = sorted(
        (path for path in machine_dir.glob("*.json") if path.name != "machine.json"),
        # merge the poorest files first so that the richest one wins
        key=lambda path: len(path.read_bytes()),
    )
    for path in files:
        data = json.loads(path.read_text())
        commit = data["commit_hash"][:HASH_LENGTH]
        result = byhash.setdefault(commit, Result(commit, data["date"]))
        result.envs.add(path.name.split("-", 1)[1].removesuffix(".json"))
        result.values.update(measurements(data))
    return sorted(byhash.values(), key=lambda result: (result.date, result.commit))


def measurements(data: dict) -> dict[tuple[str, str], float]:
    """Flatten the ``results`` of one asv file to one value per parameter set."""
    columns = data["result_columns"]
    values = {}
    for name, row in data["results"].items():
        row = dict(zip(columns, row, strict=False))
        if row["result"] is None:
            continue
        params = row.get("params") or []
        combinations = itertools.product(*params) if params else [()]
        for combination, value in zip(combinations, row["result"], strict=False):
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                values[name, ", ".join(combination)] = float(value)
    return values


def label_commits(results: list[Result], repo: str | None) -> None:
    """Name each commit after its release tag, or its branch and short hash.

    Without a pylops checkout (``--repo``) the short hash is all there is.
    """
    if repo is None:
        return
    for result in results:
        # a release commit can carry several tags (v2.7.0 and v2.7.1 do): name it
        # after the latest of them
        tags = git(repo, "tag", "--points-at", result.commit, "--sort=version:refname")
        if tags:
            result.label = tags.splitlines()[-1]
            continue
        branches = git(
            repo,
            "branch",
            "--remotes",
            "--format=%(refname:lstrip=3)",
            "--contains",
            result.commit,
        )
        for branch in ("master", "dev"):
            if branch in branches.split():
                result.label = f"{branch} {result.commit}"
                break


def git(repo: str, *args: str) -> str:
    """Run a read-only git command, returning an empty string on failure."""
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def bucket(key: tuple[str, str], baseline: float) -> str:
    """Group a benchmark with the ones the runner hardware treats alike.

    Peak memory does not depend on the runner and forms a single group; timings
    are split by order of magnitude, as the spread between runners is much
    larger for the benchmarks dominated by Python overhead than for the
    numerically heavy ones.
    """
    if not key[0].split(".")[-1].startswith("time"):
        return "peakmem"
    index = sum(baseline >= edge for edge in TIME_BUCKETS)
    return f"time{index}"


class Anchors:
    """The reference timings measured next to each commit, from ``anchors/``.

    ``anchors/<commit>.json`` holds the asv results of a fixed anchor commit,
    benchmarked on the very machine that measured ``<commit>`` (see
    ``ci/collect_anchor.py``). The anchor code never changes, so whatever moves
    its timings is the machine: subtracting them cancels the runner exactly, per
    benchmark, where the bucket medians of :class:`Normalised` can only estimate
    it from benchmarks that may themselves have changed.

    ``factor(commit, key)`` is, in log space, how much slower the anchor ran on
    that commit's runner than it usually does. It is ``None`` when the commit has
    no anchor (the history benchmarked before anchors existed) or when the anchor
    has no result for that benchmark (an operator younger than the anchor), and
    the caller falls back to the bucket estimate.
    """

    def __init__(self, directory: pathlib.Path) -> None:
        self.values: dict[str, dict[tuple[str, str], float]] = {}
        self.names: dict[str, str] = {}
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            commit = data.get("measured_for", path.stem)[:HASH_LENGTH]
            self.values[commit] = measurements(data)
            self.names[commit] = data.get("anchor", data.get("anchor_commit", "?"))
        keys = {key for values in self.values.values() for key in values}
        self.levels = {
            key: statistics.median(
                math.log(values[key])
                for values in self.values.values()
                if key in values
            )
            for key in keys
            # a level needs something to be a median of
            if sum(key in values for values in self.values.values()) > 1
        }

    def covers(self, commit: str, key: tuple[str, str]) -> bool:
        return key in self.levels and key in self.values.get(commit, {})

    def factor(self, commit: str, key: tuple[str, str]) -> float | None:
        if not self.covers(commit, key):
            return None
        return math.log(self.values[commit][key]) - self.levels[key]

    def name(self, commit: str) -> str | None:
        return self.names.get(commit)


class Normalised:
    """The measurements with the per-benchmark level and the hardware removed.

    ``residual[key][commit]`` is the deviation, in log space, of a benchmark at
    a commit from its own median level, after the median deviation of its bucket
    at that commit (what the hardware did to every benchmark of that regime) has
    been subtracted. A residual of ``log(2)`` means "twice as slow as this
    benchmark usually is, beyond what the runner explains".

    ``factors[commit][bucket]`` keeps those median deviations, i.e. how fast the
    runner of a commit was compared to the others; it is reported as is, as the
    hardware noise the rest of the report is careful to ignore.
    """

    def __init__(self, results: list[Result], anchors: Anchors | None = None) -> None:
        self.anchors = anchors
        keys = {key for result in results for key in result.values}
        levels = {
            key: statistics.median(
                math.log(result.values[key])
                for result in results
                if key in result.values
            )
            for key in keys
        }
        self.buckets = {
            key: bucket(key, math.exp(levels[key]))
            for key in keys
            # a benchmark measured once has no level to speak of
            if sum(key in result.values for result in results) > 1
        }
        self.residual: dict[tuple[str, str], dict[str, float]] = {
            key: {} for key in self.buckets
        }
        self.factors: dict[str, dict[str, float]] = {}
        for result in results:
            deviation = {
                key: math.log(value) - levels[key]
                for key, value in result.values.items()
                if key in self.buckets
            }
            factors = runner_factors(deviation, self.buckets)
            self.factors[result.commit] = factors
            for key, value in deviation.items():
                # the anchor measures this runner on this very benchmark; the
                # bucket median is the estimate to fall back on without one
                anchored = anchors.factor(result.commit, key) if anchors else None
                factor = factors[self.buckets[key]] if anchored is None else anchored
                self.residual[key][result.commit] = value - factor

    def series(self, key, results: list[Result]) -> list[tuple[Result, float]]:
        """Residuals of one benchmark over the commits that measured it."""
        byhash = self.residual[key]
        return [
            (result, byhash[result.commit])
            for result in results
            if result.commit in byhash
        ]


def runner_factors(
    deviation: dict[tuple[str, str], float], buckets: dict[tuple[str, str], str]
) -> dict[str, float]:
    """Median deviation of each bucket at one commit, i.e. the runner effect.

    Peak memory is left alone, and a bucket with too few benchmarks to have a
    robust median borrows the factor of all timings together.
    """
    grouped: dict[str, list[float]] = {}
    for key, value in deviation.items():
        grouped.setdefault(buckets[key], []).append(value)
    overall = [
        value
        for name, values in grouped.items()
        if name != "peakmem"
        for value in values
    ]
    fallback = statistics.median(overall) if overall else 0.0
    return {
        name: 0.0
        if name == "peakmem"
        else (statistics.median(values) if len(values) >= MIN_BUCKET else fallback)
        for name, values in grouped.items()
    }


class Hardware:
    """What measured each commit, from ``runners/<commit>.json``.

    Commits are grouped into classes of hardware that are *comparable*: same
    CPU model, same answer to "is AVX-512 available", same number of cores and
    the same measurement setup (threads pinned to one, benchmarks pinned to one
    core). Anything else and a ratio between two commits measures the machines
    as much as the code.

    The class of a commit benchmarked before the descriptions were recorded, or
    of one whose file is missing, is ``None``: unknown hardware is never used to
    dismiss a step, it just cannot be used to confirm one either.
    """

    # instruction sets worth distinguishing: same CPU model, wildly different
    # timings depending on whether the VM exposes them
    FLAGS = ("avx512f", "avx2")

    def __init__(self, directory: pathlib.Path, machine: str) -> None:
        self.machine = machine
        self.descriptions: dict[str, dict] = {}
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                description = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            commit = description.get("commit", path.stem)[:HASH_LENGTH]
            self.descriptions[commit] = description

    def describe(self, commit: str) -> dict | None:
        return self.descriptions.get(commit)

    def klass(self, commit: str) -> str | None:
        """Identity of the hardware, as far as comparability goes."""
        description = self.descriptions.get(commit)
        if description is None:
            return None
        flags = set(description.get("flags", ()))
        sets = "+".join(flag for flag in self.FLAGS if flag in flags) or "baseline"
        threads = sorted(set(description.get("threads", {}).values()))
        setup = (
            f"{','.join(threads) or '?'}thr/core{description.get('taskset_cpu', '?')}"
        )
        return f"{description.get('cpu', '?')} [{sets}] {description.get('num_cpu', '?')}cpu {setup}"

    def comparable(self, one: list[str], other: list[str]) -> bool:
        """Whether two groups of commits were measured on the same hardware.

        Unknown hardware is comparable with anything: without a description
        there is nothing to object with.
        """
        classes = {self.klass(commit) for commit in one} - {None}
        others = {self.klass(commit) for commit in other} - {None}
        return not classes or not others or bool(classes & others)

    def table(self, results: list["Result"], anchors: "Anchors") -> list[str]:
        """Markdown table of the hardware behind each commit."""
        lines = [
            "| commit | CPU | instruction sets | cores | threads | pinned to | anchor |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for result in results:
            anchor = anchors.name(result.commit) or "_none_"
            description = self.describe(result.commit)
            if description is None:
                lines.append(
                    f"| {result.label} | _not recorded_ | - | - | - | - | {anchor} |"
                )
                continue
            flags = set(description.get("flags", ()))
            sets = ", ".join(f for f in self.FLAGS if f in flags) or "baseline"
            threads = sorted(set(description.get("threads", {}).values())) or ["?"]
            lines.append(
                f"| {result.label} | {description.get('cpu', '?')} | {sets} "
                f"| {description.get('num_cpu', '?')} | {', '.join(threads)} "
                f"| core {description.get('taskset_cpu', '?')} | {anchor} |"
            )
        return lines


class Step:
    """A sustained change of level in the series of one benchmark."""

    def __init__(self, key, ratio: float, since: Result, before, after) -> None:
        self.key = key
        self.ratio = ratio
        self.since = since
        self.before = before  # commits at the old level
        self.after = after  # commits at the new level, `since` first
        # set by `find_steps` once the hardware of the two levels is known: a
        # step that lines up with a change of machine is not evidence of either
        self.confounded = False
        self.reproduced = len(after) >= 2

    @property
    def confirmed(self) -> bool:
        """Reproduced on a second commit, and not explained by the hardware."""
        return self.reproduced and not self.confounded

    @property
    def name(self) -> str:
        return f"{self.key[0]}({self.key[1]})" if self.key[1] else self.key[0]


def find_step(key, series: list[tuple[Result, float]], limit: float) -> Step | None:
    """Cleanest sustained level change in a benchmark's residuals, if any.

    Every split of the series is a candidate step, kept only when the two levels
    it separates do not overlap at all: for a slowdown, every residual after the
    split must be above every residual before it. Candidates are ranked by the
    *gap* between the levels rather than by the distance between their medians,
    so that the step is attributed to the commit where the benchmark actually
    moved and not to a later one that happens to sit slightly higher.

    A step qualifies when the medians are ``limit`` apart and the gap itself
    covers a third of it. A merely noisy benchmark can drift far enough for the
    medians to differ, but its levels then nearly touch, which the gap rejects.
    Only slowdowns are looked for: this is a regression report.
    """
    min_gap = math.log(1 + (limit - 1) / 3)
    best = None
    for split in range(MIN_BASELINE, len(series)):
        before = [value for _, value in series[:split]]
        after = [value for _, value in series[split:]]
        delta = statistics.median(after) - statistics.median(before)
        gap = min(after) - max(before)
        if gap < min_gap or delta < math.log(limit):
            continue
        if best is None or gap > best[0]:
            best = (gap, delta, split)
    if best is None:
        return None
    _, delta, split = best
    return Step(
        key,
        math.exp(delta),
        series[split][0],
        [result for result, _ in series[:split]],
        [result for result, _ in series[split:]],
    )


def find_steps(
    results: list[Result],
    normalised: Normalised,
    threshold: float,
    memory_threshold: float,
    hardware: Hardware,
    anchors: Anchors | None = None,
) -> list[Step]:
    """Every regression above the threshold of its unit, largest first.

    A step is marked as confounded when no commit before it shares its hardware
    with a commit after it: the machine changed exactly where the timing did,
    and the two explanations cannot be told apart. Peak memory is exempt, as
    allocations do not depend on the machine.
    """
    steps = []
    for key in normalised.residual:
        series = normalised.series(key, results)
        if len(series) <= MIN_BASELINE:
            continue
        memory = bucket(key, 0.0) == "peakmem"
        step = find_step(key, series, memory_threshold if memory else threshold)
        if step is None:
            continue
        if not memory:
            # with an anchor on both sides the runner is already cancelled, so
            # the hardware cannot be the explanation of what is left
            anchored = anchors is not None and all(
                anchors.covers(result.commit, key)
                for result in step.before + step.after
            )
            step.confounded = not anchored and not hardware.comparable(
                [result.commit for result in step.before],
                [result.commit for result in step.after],
            )
        steps.append(step)
    return sorted(steps, key=lambda step: -step.ratio)


def find_lost(results: list[Result]) -> list[tuple[str, str]]:
    """Benchmarks that had results but have none in the newest commit.

    asv writes ``null`` both for a benchmark that errored and for one that was
    skipped (a missing optional dependency, or an operator that does not exist
    in the benchmarked release), so this is a hint to go and read the log of the
    benchmark workflow, not a failure by itself. Only benchmarks that succeeded
    in the two preceding commits are reported, to stay quiet about the ones
    that come and go with the release being benchmarked.
    """
    newest, *previous = results[::-1]
    if len(previous) < 2:
        return []
    return sorted(
        key
        for key in previous[0].values.keys() & previous[1].values.keys()
        if key not in newest.values
    )


def format_value(key: tuple[str, str], value: float) -> str:
    """Format one measurement in the unit of its benchmark."""
    if bucket(key, 0.0) == "peakmem":
        return f"{value / 1024**2:.1f} MB"
    if value < 1e-3:
        return f"{value * 1e6:.0f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.2f} s"


def timeline(step: Step, results: list[Result]) -> str:
    """Values of a flagged benchmark over the most recent commits."""
    shown = [result for result in results[-TIMELINE:] if step.key in result.values]
    return " → ".join(
        f"**{format_value(step.key, result.values[step.key])}**"
        if result in step.after
        else format_value(step.key, result.values[step.key])
        for result in shown
    )


def table(steps: list[Step], results: list[Result]) -> list[str]:
    """Markdown table of flagged benchmarks."""
    lines = [
        "| benchmark | slowdown | since | recent values |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{step.name}` | {step.ratio:.2f}x | {step.since.label} "
        f"| {timeline(step, results)} |"
        for step in steps
    ]
    return lines


def report(
    results: list[Result],
    normalised: Normalised,
    steps: list[Step],
    lost: list[tuple[str, str]],
    hardware: Hardware,
    anchors: Anchors,
    args: argparse.Namespace,
) -> tuple[str, bool]:
    """Render the markdown report, and whether it holds a confirmed regression."""
    confirmed = [step for step in steps if step.confirmed]
    confounded = [step for step in steps if step.confounded]
    suspected = [step for step in steps if not step.confirmed and not step.confounded]
    newest = results[-1]
    lines = [
        "# asv regression report",
        "",
        f"{len(results)} commits benchmarked on `{args.machine}`, "
        f"{results[0].label} → {newest.label}. A benchmark is flagged when its "
        f"timing steps up by {args.threshold}x "
        f"({args.memory_threshold}x for peak memory) and stays there, after the "
        "hardware differences between runners have been normalised out.",
        "",
        f"{sum(anchors.name(result.commit) is not None for result in results)} of "
        f"them carry an anchor measurement, which cancels their runner exactly "
        f"instead of estimating it.",
        "",
    ]

    if confirmed:
        lines += [
            f"## ⚠️ {len(confirmed)} confirmed regression(s)",
            "",
            "Reproduced on at least two runners, so this is unlikely to be noise.",
            "",
            *table(confirmed, results),
            "",
        ]
    else:
        lines += ["## ✅ No confirmed regression", ""]

    if confounded:
        lines += [
            f"## 🎲 {len(confounded)} step(s) the hardware explains as well",
            "",
            "Every commit before the step was measured on different hardware "
            "from every commit after it, so the change of level cannot be "
            "attributed to the code: benchmarking one of these commits again, "
            "on a runner of the other kind, is what settles it. The hardware of "
            "each commit is at the bottom of this report.",
            "",
            *table(confounded, results),
            "",
        ]

    if suspected:
        lines += [
            f"## 🔍 {len(suspected)} unconfirmed step(s) at {newest.label}",
            "",
            "Only measured on one runner so far: the next nightly run will tell "
            "whether it is a regression or an unlucky run.",
            "",
            *table(suspected, results),
            "",
        ]

    if lost:
        lines += [
            f"## 🚫 {len(lost)} benchmark(s) without a result at {newest.label}",
            "",
            "They succeeded in the two preceding commits. asv reports an errored "
            "and a skipped benchmark alike, so check the log of the "
            "[PyLops-benchmarks run]"
            "(../../actions/workflows/benchmarks.yaml) that produced "
            f"{newest.label}.",
            "",
            *(
                f"- `{name}({params})`" if params else f"- `{name}`"
                for name, params in lost
            ),
            "",
        ]

    lines += [
        "<details><summary>Hardware that measured each commit</summary>",
        "",
        "Recorded by `ci/record_runner.py` in `runners/`. Two runners reporting "
        "the same CPU model do not necessarily offer the same instruction sets: "
        "AVX-512 is masked on some of them, which is worth a factor of three to "
        "six on the numerically heavy benchmarks and is why the table spells the "
        "instruction sets out.",
        "",
        *hardware.table(results, anchors),
        "</details>",
        "",
        "<details><summary>Runner spread between the benchmarked commits</summary>",
        "",
        "Timings of the same code on the GitHub-hosted runners, relative to the "
        "median commit, per timing regime. This is the hardware noise the report "
        "normalises out; it is also why raw numbers on the "
        "[website](https://pylops.github.io/pylops-asv/) jump around.",
        "",
        *spread(results, normalised),
        "</details>",
        "",
    ]
    return "\n".join(lines), bool(confirmed)


def spread(results: list[Result], normalised: Normalised) -> list[str]:
    """Markdown table of the runner factors, per commit and timing regime."""
    names = sorted({name for name in normalised.buckets.values() if name != "peakmem"})
    headers = {
        "time0": "< 0.1 ms",
        "time1": "0.1-1 ms",
        "time2": "1-10 ms",
        "time3": "10-100 ms",
        "time4": "> 100 ms",
    }
    lines = [
        "| commit | " + " | ".join(headers.get(name, name) for name in names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for result in results:
        factors = normalised.factors[result.commit]
        lines.append(
            f"| {result.label} | "
            + " | ".join(
                f"{math.exp(factors[name]):.2f}x" if name in factors else "-"
                for name in names
            )
            + " |"
        )
    return lines


def annotate(steps: list[Step]) -> None:
    """Warn about each confirmed regression in the GitHub Actions log.

    The annotations appear at the top of the run page, next to the report
    itself; the run stays green, as a regression is something to look at rather
    than a broken build.
    """
    for step in steps:
        if step.confirmed:
            print(
                f"::warning title=Performance regression::{step.name} is "
                f"{step.ratio:.2f}x slower since {step.since.label}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results", help="directory with the committed asv results"
    )
    parser.add_argument(
        "--machine", default="gha-ubuntu-latest", help="machine whose results to read"
    )
    parser.add_argument(
        "--runners",
        default="runners",
        help="directory with the hardware description of each benchmarked commit",
    )
    parser.add_argument(
        "--anchors",
        default="anchors",
        help="directory with the anchor measurement taken next to each commit",
    )
    parser.add_argument(
        "--repo", help="pylops checkout, used to name commits after their release tag"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="slowdown factor a timing must reach to be flagged (default: 1.5)",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=1.2,
        help="same, for peak memory (default: 1.2, allocations are not noisy)",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("a"),
        default=sys.stdout,
        help="file the markdown report is appended to (default: stdout)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="emit a GitHub Actions warning annotation per confirmed regression",
    )
    args = parser.parse_args(argv)

    results = read_results(pathlib.Path(args.results), args.machine)
    if len(results) <= MIN_BASELINE:
        msg = (
            f"only {len(results)} commit(s) with results for {args.machine!r}: "
            f"at least {MIN_BASELINE + 1} are needed to detect a regression"
        )
        raise SystemExit(msg)
    label_commits(results, args.repo)

    hardware = Hardware(pathlib.Path(args.runners), args.machine)
    anchors = Anchors(pathlib.Path(args.anchors))
    normalised = Normalised(results, anchors)
    steps = find_steps(
        results, normalised, args.threshold, args.memory_threshold, hardware, anchors
    )
    text, _ = report(
        results, normalised, steps, find_lost(results), hardware, anchors, args
    )
    print(text, file=args.output)
    if args.output is not sys.stdout:
        print(text)
    if args.annotate:
        annotate(steps)
    # a regression is a warning, never a failure: only a broken report fails
    return 0


if __name__ == "__main__":
    sys.exit(main())
