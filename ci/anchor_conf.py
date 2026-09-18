"""Write the asv configuration used to benchmark the anchor commit.

Used by ``.github/workflows/benchmarks.yaml``. Every job benchmarks, besides
the commit it was given, a fixed *anchor* commit, on the same runner and in the
same conditions. The two measurements share everything the machine contributes,
so the ratio between them is comparable across runners, which raw timings are
not: the runners differ in speed and, worse, in the instruction sets they
expose, which is worth a factor of three to six on the numerically heavy
benchmarks.

The anchor's measurements must not land in ``results/``: asv keys a result file
on the commit it benchmarked, so the anchor would overwrite its own entry in the
published series over and over, once per job, each time with the timings of a
different machine. This configuration is therefore a copy of ``asv.conf.json``
with ``results_dir`` pointing into ``.asv``. ``env_dir`` is left where it is so
that the environment built for the benchmarked commit is reused rather than
recreated: only the anchor's wheel has to be built.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASV_DIR = ROOT / ".asv"
ANCHOR_CONF = ASV_DIR / "asv.anchor.conf.json"
RESULTS_DIR = "anchor-results"


def load_conf(path: pathlib.Path) -> dict:
    """Load an asv configuration file (JSON with ``//`` comments)."""
    text = re.sub(r"^\s*//.*$", "", path.read_text(), flags=re.MULTILINE)
    return json.loads(text)


def main() -> int:
    conf = load_conf(ROOT / "asv.conf.json")
    # paths are relative to the directory containing the configuration file
    conf["benchmark_dir"] = "../benchmarks"
    conf["env_dir"] = "env"
    conf["results_dir"] = RESULTS_DIR
    conf["html_dir"] = "anchor-html"
    ASV_DIR.mkdir(exist_ok=True)
    ANCHOR_CONF.write_text(json.dumps(conf, indent=4) + "\n")
    print(ANCHOR_CONF.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
