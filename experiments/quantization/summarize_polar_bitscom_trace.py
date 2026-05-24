import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


MARKERS: List[Tuple[str, re.Pattern[str]]] = [
    ("evidence_polarparallel_init", re.compile(r"\[trace-evidence.*component=PolarParallel.*action=init")),
    ("evidence_stage_partition", re.compile(r"\[trace-evidence.*component=PolarParallel.*action=stage_partition")),
    ("evidence_polar_hook_trigger", re.compile(r"\[trace-evidence.*component=POLAR.*action=hook_trigger")),
    ("evidence_polar_to_bitscom", re.compile(r"\[trace-evidence.*component=POLAR.*action=handoff_to_bitscom_lowbit")),
    ("evidence_polar_wait", re.compile(r"\[trace-evidence.*component=POLAR.*action=wait_for_async_dp_reduce")),
    ("evidence_polar_error_feedback", re.compile(r"\[trace-evidence.*component=POLAR.*action=error_feedback_update")),
    ("evidence_bitscom_entry", re.compile(r"\[trace-evidence.*component=bitscom.*action=all_reduce_entry")),
    ("evidence_bitscom_path", re.compile(r"\[trace-evidence.*component=bitscom.*action=path_selected")),
    ("evidence_bitscom_plan", re.compile(r"\[trace-evidence.*component=bitscom.*action=lowbit_allreduce_plan|\[trace-evidence.*component=bitscom.*action=pipeline_a_plan")),
    ("evidence_bitscom_quantize", re.compile(r"\[trace-evidence.*component=bitscom.*action=quantize_pack")),
    ("evidence_bitscom_collective", re.compile(r"\[trace-evidence.*component=bitscom.*action=collective")),
    ("evidence_bitscom_done", re.compile(r"\[trace-evidence.*component=bitscom.*action=lowbit_allreduce_done")),
    ("polar_parallel_active", re.compile(r"\[PolarParallel\]|PolarParallel:init")),
    ("polar_step_schedule", re.compile(r"\[polar-step-debug.*schedule\.step enter")),
    ("polar_hook_trigger", re.compile(r"\[polar-hook-debug.*trigger")),
    ("polar_hook_lowbit_allreduce", re.compile(r"\[polar-hook-debug.*lowbit all_reduce start")),
    ("bitscom_allreduce_entry", re.compile(r"\[bitscom-debug.*all_reduce entry")),
    ("bitscom_path_selected", re.compile(r"\[bitscom-debug.*path=")),
    ("bitscom_pipeline_phase", re.compile(r"\[bitscom-debug.*pipeline_a .*phase")),
    ("bitscom_lowbit_allreduce", re.compile(r"\[bitscom-debug.*lowbit allreduce start")),
    ("bitscom_quantize", re.compile(r"\[bitscom-debug.*quantize shard .* start")),
    ("bitscom_collective", re.compile(r"\[bitscom-debug.*all_to_all packed start|\[bitscom-debug.*all_gather packed start")),
]


def expand_paths(paths: Iterable[Path]) -> List[Path]:
    expanded: List[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.is_file():
            expanded.append(path)
        else:
            print(f"[warn] skipping missing path: {path}")
    return expanded


def iter_lines(paths: Iterable[Path]):
    for path in expand_paths(paths):
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
