"""Write an asv configuration for local runs against the installed pylops.

``asv run --python=same`` refuses a revision range and otherwise benchmarks the
head of every branch listed in ``asv.conf.json`` (i.e. the suite would run
twice, once for ``master`` and once for ``dev``). This helper copies the main
configuration into ``.asv/asv.local.conf.json`` with

* ``repo`` pointing to a local checkout of pylops (``$PYLOPS_REPO``, default
  ``../pylops``) instead of GitHub,
* ``branches`` replaced by ``HEAD`` so that ``make bench`` runs the suite
  exactly once for that checkout,
* ``results_dir`` moved to ``.asv/results-local`` so that local numbers never
  end up in the tracked ``results`` directory.
"""

import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASV_DIR = ROOT / ".asv"
LOCAL_CONF = ASV_DIR / "asv.local.conf.json"


def load_conf(path: pathlib.Path) -> dict:
    """Load an asv configuration file (JSON with ``//`` comments)."""
    text = re.sub(r"^\s*//.*$", "", path.read_text(), flags=re.MULTILINE)
    return json.loads(text)


def main() -> int:
    repo = pathlib.Path(os.environ.get("PYLOPS_REPO", ROOT.parent / "pylops"))
    repo = repo.expanduser().resolve()
    if not (repo / ".git").exists():
        print(
            f"{repo} is not a git checkout of pylops (set PYLOPS_REPO)", file=sys.stderr
        )
        return 1
    conf = load_conf(ROOT / "asv.conf.json")
    conf["repo"] = str(repo)
    conf["branches"] = ["HEAD"]
    # paths are relative to the directory containing the configuration file
    conf["benchmark_dir"] = "../benchmarks"
    conf["env_dir"] = "env"
    conf["results_dir"] = "results-local"
    conf["html_dir"] = "html"
    ASV_DIR.mkdir(exist_ok=True)
    LOCAL_CONF.write_text(json.dumps(conf, indent=4) + "\n")
    print(LOCAL_CONF.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
