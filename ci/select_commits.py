"""Resolve the pylops commits to benchmark in CI to a JSON list of full hashes.

Used by ``.github/workflows/benchmarks.yaml``, which checks out pylops in a
sub-directory passed with ``--repo``:

* on ``schedule`` the heads of the configured branches plus the latest
  release tag (``--last-tags 1``);
* on ``workflow_dispatch`` either the last ``N`` release tags plus the heads
  of the configured branches (``--last-tags N``), a single commit (``--sha``)
  or an arbitrary revision range understood by ``git rev-list``
  (``--revisions``).

With ``--skip-benchmarked RESULTS_DIR`` commits that already have a results
file in ``RESULTS_DIR/<machine>/`` are dropped (an empty list is then a valid
outcome).

Tags are matched case-insensitively (the repository has a ``V2.4.0`` tag) and
resolved through ``refs/tags/`` to avoid clashes with branch names.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

BRANCHES = ("master", "dev")
TAG_PATTERN = re.compile(r"^v\d+\.\d+", re.IGNORECASE)
HASH_LENGTH = 8  # must match "hash_length" in asv.conf.json


def git(repo: str, *args: str) -> str:
    return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()


def resolve(repo: str, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def branch_heads(repo: str) -> list[str]:
    heads = []
    for branch in BRANCHES:
        for ref in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
            try:
                heads.append(resolve(repo, ref))
                break
            except subprocess.CalledProcessError:
                continue
    return heads


def last_tags(repo: str, n: int) -> list[str]:
    tags = [
        tag
        for tag in git(repo, "tag", "--list", "--sort=-creatordate").splitlines()
        if TAG_PATTERN.match(tag)
    ]
    return [resolve(repo, f"refs/tags/{tag}") for tag in tags[:n]]


def revisions(repo: str, spec: str) -> list[str]:
    return git(repo, "rev-list", "--reverse", spec).splitlines()


def benchmarked(results_dir: pathlib.Path) -> set[str]:
    """Short hashes of the commits with at least one results file."""
    return {
        path.name.split("-", 1)[0]
        for path in results_dir.glob("*/*.json")
        if path.name != "machine.json"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="path to the pylops checkout")
    parser.add_argument(
        "--skip-benchmarked",
        metavar="RESULTS_DIR",
        help="drop commits that already have results in RESULTS_DIR",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sha", help="single commit to benchmark")
    group.add_argument(
        "--last-tags", type=int, help="last N release tags + branch heads"
    )
    group.add_argument("--revisions", help="git rev-list range, e.g. v2.7.0..master")
    args = parser.parse_args(argv)

    if args.sha:
        commits = [resolve(args.repo, args.sha)]
    elif args.last_tags is not None:
        commits = last_tags(args.repo, args.last_tags) + branch_heads(args.repo)
    else:
        commits = revisions(args.repo, args.revisions)

    # de-duplicate while preserving order
    commits = list(dict.fromkeys(commits))
    if not commits:
        print("no commits selected", file=sys.stderr)
        return 1
    if args.skip_benchmarked:
        done = benchmarked(pathlib.Path(args.skip_benchmarked))
        commits = [c for c in commits if c[:HASH_LENGTH] not in done]
    print(json.dumps(commits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
