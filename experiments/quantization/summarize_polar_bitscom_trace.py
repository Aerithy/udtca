import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


MARKERS: List[Tuple[str, re.Pattern[str]]] = [
    ("polar_parallel_active", re.compile(r"\[PolarParallel\]|PolarParallel:init")),
    ("polar_step_schedule", re.compile(r"\[polar-step-debug.*schedule\.step enter")),
    ("polar_hook_trigger", re.compile(r"\[polar-hook-debug.*trigger")),
    ("polar_hook_lowbit_allreduce", re.compile(r"\[polar-hook-debug.*lowbit all_reduce start")),
    ("bitscom_lowbit_allreduce", re.compile(r"\[bitscom-debug.*lowbit allreduce start")),
    ("bitscom_quantize", re.compile(r"\[bitscom-debug.*quantize shard .* start")),
    ("bitscom_collective", re.compile(r"\[bitscom-debug.*all_to_all packed start|\[bitscom-debug.*all_gather packed start")),
]


def iter_lines(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                yield path, line_no, line.rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract defense-friendly evidence from POLAR+bitscom trace logs."
    )
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--max-lines-per-marker", type=int, default=8)
    args = parser.parse_args()

    found = {name: [] for name, _ in MARKERS}
    for path, line_no, line in iter_lines(args.logs):
        for name, pattern in MARKERS:
            if len(found[name]) >= args.max_lines_per_marker:
                continue
            if pattern.search(line):
                found[name].append((path, line_no, line))

    for name, entries in found.items():
        print(f"\n## {name}")
        if not entries:
            print("MISSING")
            continue
        for path, line_no, line in entries:
            print(f"{path}:{line_no}: {line}")


if __name__ == "__main__":
    main()
