"""Record the hardware that benchmarked one pylops commit.

Used by ``.github/workflows/benchmarks.yaml``, which writes one
``runners/<commit>.json`` per benchmarked commit, next to the asv results.

Every commit is benchmarked in its own matrix job, hence on a different
GitHub-hosted runner, and the runners are not identical: the same benchmark can
be twice as slow on one CPU model as on another. The CPU therefore has to be
known to tell a real regression from a change of hardware, but it deliberately
does *not* go into asv's own ``machine.json``: asv groups results into one graph
per distinct set of machine parameters, so recording the CPU there would split
the published series into one line per runner model. It is kept here instead, so
that ``results/`` stays a single continuous series while the hardware behind
every point remains known.

The file is overwritten when a commit is benchmarked again, and therefore always
describes the runner that produced the results currently committed.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess

# Environment variables pinning the number of threads of the numerical
# libraries, recorded along with the hardware: timings measured with a
# different setting are not comparable, whatever the CPU.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
# Fields of `lscpu -J`, mapped to the name they are recorded under
LSCPU_FIELDS = {
    "Model name": "cpu",
    "Vendor ID": "vendor",
    "CPU family": "cpu_family",
    "Model": "cpu_model",
    "Stepping": "stepping",
    "CPU max MHz": "cpu_max_mhz",
    "BogoMIPS": "bogomips",
    # `lscpu -J` nests the caches under "Caches (sum of all)" and drops the
    # "cache" suffix that the plain text output uses: accept both spellings
    "L1d": "l1d_cache",
    "L1d cache": "l1d_cache",
    "L2": "l2_cache",
    "L2 cache": "l2_cache",
    "L3": "l3_cache",
    "L3 cache": "l3_cache",
    "Flags": "flags",
    "Hypervisor vendor": "hypervisor",
}


def _lscpu() -> dict:
    """CPU description, as the fields of ``lscpu`` that identify the model.

    ``lscpu -J`` nests the fields under ``children``, with a trailing colon in
    the field names. Anything missing (a field the runner's ``lscpu`` does not
    report) is simply left out.
    """
    try:
        out = subprocess.run(
            ["lscpu", "-J"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}

    flat = {}

    def walk(entries: list) -> None:
        for entry in entries:
            field = entry.get("field", "").rstrip(":")
            if "data" in entry:
                flat[field] = entry["data"]
            walk(entry.get("children", []))

    walk(json.loads(out).get("lscpu", []))
    cpu = {name: flat[field] for field, name in LSCPU_FIELDS.items() if field in flat}
    # The flags name the instruction sets available (avx512 and the like), which
    # is what makes two CPU models differ on the numerically heavy benchmarks:
    # sorted, so that two runners can be compared with a plain diff.
    if "flags" in cpu:
        cpu["flags"] = sorted(cpu["flags"].split())
    return cpu


def _meminfo_mb() -> int | None:
    """Total RAM in MiB, from ``/proc/meminfo``."""
    try:
        text = pathlib.Path("/proc/meminfo").read_text()
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+) kB$", text, flags=re.MULTILINE)
    return int(match.group(1)) // 1024 if match else None


def _uname() -> dict:
    """Kernel and architecture, as reported by ``uname``."""
    try:
        out = subprocess.run(
            ["uname", "-srm"], capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return dict(zip(("os", "kernel", "arch"), out, strict=False))


def describe(commit: str, machine: str, taskset_cpu: str | None, env: dict) -> dict:
    """Everything known about the machine that benchmarked ``commit``."""
    info = {
        "commit": commit,
        "machine": machine,
        "benchmarked_at": datetime.datetime.now(datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "num_cpu": os.cpu_count(),
        "ram_mb": _meminfo_mb(),
        "taskset_cpu": taskset_cpu or None,
        "threads": {var: env[var] for var in THREAD_VARS if var in env},
    }
    info.update(_uname())
    info.update(_lscpu())
    # the run that produced it, to find the log of a suspicious point back
    for key, name in (
        ("GITHUB_RUN_ID", "run_id"),
        ("GITHUB_RUN_ATTEMPT", "run_attempt"),
    ):
        if env.get(key):
            info[name] = env[key]
    return {key: value for key, value in info.items() if value is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="benchmarked pylops commit")
    parser.add_argument("--machine", required=True, help="asv machine name")
    parser.add_argument(
        "--taskset-cpu", default=None, help="core the benchmarks are pinned to"
    )
    parser.add_argument(
        "--output", required=True, type=pathlib.Path, help="json file to write"
    )
    args = parser.parse_args()

    info = describe(args.commit, args.machine, args.taskset_cpu, dict(os.environ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    # the log of the job is the first place one looks at a surprising timing
    summary = {
        key: info[key]
        for key in ("cpu", "num_cpu", "ram_mb", "taskset_cpu")
        if key in info
    }
    print(f"{args.output}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
