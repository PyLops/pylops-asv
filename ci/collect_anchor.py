"""Store the anchor measurement of one job as ``anchors/<commit>.json``.

Used by ``.github/workflows/benchmarks.yaml``, after the anchor commit has been
benchmarked on the same runner as the commit the job was given (see
``ci/anchor_conf.py`` for why).

The file is the asv result file of the *anchor*, named after the commit whose
job measured it: ``anchors/<commit>.json`` answers "how fast was the reference
code on the machine that measured <commit>", which is what turns two timings
taken on different runners into a comparison of the code. It keeps asv's own
format so that ``ci/regression_report.py`` reads it with the same parser as the
results.

When the job's own commit is the anchor, its results are its anchor: asv has
already measured exactly that, and running it twice would only add noise.
"""

import argparse
import json
import pathlib
import sys

HASH_LENGTH = 8  # must match "hash_length" in asv.conf.json


def newest_result(directory: pathlib.Path) -> pathlib.Path:
    """The asv result file in ``directory``, ignoring ``machine.json``.

    Only the anchor is ever benchmarked with the anchor configuration, so there
    is normally a single file; should a stale one survive a re-run, the most
    recently written wins.
    """
    files = [path for path in directory.glob("*.json") if path.name != "machine.json"]
    if not files:
        msg = f"no asv result file in {directory}"
        raise SystemExit(msg)
    return max(files, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        type=pathlib.Path,
        help="machine directory holding the anchor's asv results",
    )
    parser.add_argument("--commit", required=True, help="the commit this job measured")
    parser.add_argument("--anchor", required=True, help="the anchor revision, as given")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    result = json.loads(newest_result(args.results).read_text())
    anchor_commit = result.get("commit_hash", "")
    if anchor_commit[:HASH_LENGTH] == args.commit[:HASH_LENGTH]:
        print(
            f"note: {args.commit[:HASH_LENGTH]} is the anchor itself",
            file=sys.stderr,
        )
    # what the file is about, for anyone reading it without this script at hand
    result["anchor"] = args.anchor
    result["anchor_commit"] = anchor_commit
    result["measured_for"] = args.commit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(f"{args.output}: anchor {args.anchor} <{anchor_commit[:HASH_LENGTH]}>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
