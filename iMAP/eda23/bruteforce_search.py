#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FPGA_STATS_RE = re.compile(r"Stats of FPGA:.*?area=(\d+), depth=(\d+)")


@dataclass(frozen=True)
class Action:
    name: str
    command: str


@dataclass(frozen=True)
class Candidate:
    steps: tuple[Action, ...]
    map_action: Action


@dataclass
class EvalResult:
    candidate: Candidate
    cost: float
    area: int
    depth: int
    stdout: str
    stderr: str


DEFAULT_ACTIONS = (
    Action("balance", "balance"),
    Action("rewrite", "rewrite"),
    Action("rewrite_lp", "rewrite -l"),
    Action("rewrite_zg", "rewrite -z"),
    Action("rewrite_lpz", "rewrite -l -z"),
    Action("refactor", "refactor"),
    Action("refactor_lp", "refactor -l"),
    Action("refactor_zg", "refactor -z"),
    Action("refactor_lpz", "refactor -l -z"),
    Action("lut_opt", "lut_opt"),
    Action("lut_opt_zg", "lut_opt -z"),
)


DEFAULT_MAP_ACTIONS = (
    Action("map_default", "map_fpga"),
    Action("map_cut12", "map_fpga -P 12 -C 6 -G 1 -L 2"),
    Action("map_area", "map_fpga -P 12 -C 6 -G 2 -L 3"),
    Action("map_area_mid", "map_fpga -P 10 -C 6 -G 2 -L 2"),
)


def candidate_count(num_actions: int, num_maps: int, max_steps: int) -> int:
    return sum((num_actions ** depth) * num_maps for depth in range(max_steps + 1))


def generate_candidates(
    actions: tuple[Action, ...],
    map_actions: tuple[Action, ...],
    max_steps: int,
) -> Iterable[Candidate]:
    for depth in range(max_steps + 1):
        for steps in itertools.product(actions, repeat=depth):
            for map_action in map_actions:
                yield Candidate(steps=steps, map_action=map_action)


def build_sequence(
    candidate: Candidate,
    use_history: bool,
    history_capacity: int = 5,
) -> list[str]:
    commands: list[str] = []
    history_size = 0

    if use_history:
        commands.extend(["history -c", "history -a"])
        history_size = 1

    for step in candidate.steps:
        commands.append(step.command)
        if use_history and history_size < history_capacity:
            commands.append("history -a")
            history_size += 1

    map_cmd = candidate.map_action.command
    if use_history and history_size >= 2:
        map_cmd = f"{map_cmd} -t 1"
    commands.append(map_cmd)
    return commands


def serialize_sequence(commands: list[str]) -> str:
    return "; ".join(commands) + ";"


def evaluate_candidate(
    aig_path: Path,
    imap_bin: Path,
    candidate: Candidate,
    use_history: bool,
    timeout: float | None,
) -> EvalResult | None:
    commands = [f"read_aiger -f {aig_path}"]
    commands.extend(build_sequence(candidate, use_history=use_history))
    commands.append("print_stats -t 1")
    flow = serialize_sequence(commands)

    proc = subprocess.run(
        [str(imap_bin), "-c", flow],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return None

    match = FPGA_STATS_RE.search(stdout)
    if not match:
        return None

    area = int(match.group(1))
    depth = int(match.group(2))
    cost = 0.6 * depth + 0.4 * area
    return EvalResult(
        candidate=candidate,
        cost=cost,
        area=area,
        depth=depth,
        stdout=stdout,
        stderr=stderr,
    )


def write_sequence_file(path: Path, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sequence + "\n", encoding="utf-8")


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def candidate_to_labels(candidate: Candidate) -> list[str]:
    return [step.name for step in candidate.steps] + [candidate.map_action.name]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Brute-force flow search for the EDA23 iMAP problem."
    )
    parser.add_argument("input_aig", type=Path)
    parser.add_argument("output_seq", type=Path)
    parser.add_argument(
        "--imap-bin",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bin" / "imap",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("BF_MAX_STEPS", "3")),
        help="Enumerate optimization sequences with lengths 0..N.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=(
            float(os.environ["BF_EVAL_TIMEOUT"])
            if "BF_EVAL_TIMEOUT" in os.environ
            else None
        ),
        help="Optional per-candidate timeout in seconds.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history-based mapping candidates.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs.",
    )
    args = parser.parse_args()

    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")

    input_aig = args.input_aig.resolve()
    output_seq = args.output_seq.resolve()
    imap_bin = args.imap_bin.resolve()
    use_history = not args.no_history

    if not input_aig.is_file():
        raise SystemExit(f"input AIG not found: {input_aig}")
    if not imap_bin.is_file():
        raise SystemExit(f"imap binary not found: {imap_bin}")

    total = candidate_count(len(DEFAULT_ACTIONS), len(DEFAULT_MAP_ACTIONS), args.max_steps)
    if not args.quiet:
        print(
            f"searching {input_aig.name}: {total} candidates "
            f"(max_steps={args.max_steps}, use_history={use_history})",
            file=sys.stderr,
        )

    start = time.time()
    evaluated = 0
    best: EvalResult | None = None

    for candidate in generate_candidates(DEFAULT_ACTIONS, DEFAULT_MAP_ACTIONS, args.max_steps):
        result = evaluate_candidate(
            input_aig,
            imap_bin,
            candidate,
            use_history=use_history,
            timeout=args.timeout,
        )
        if result is None:
            continue

        evaluated += 1
        if (
            best is None
            or result.cost < best.cost
            or (result.cost == best.cost and (result.depth, result.area) < (best.depth, best.area))
        ):
            best = result
            if not args.quiet:
                print(
                    f"best update: cost={best.cost:.3f} depth={best.depth} area={best.area} "
                    f"flow={candidate_to_labels(best.candidate)}",
                    file=sys.stderr,
                )

    if best is None:
        fallback = "rewrite; balance; refactor; lut_opt; map_fpga;"
        write_sequence_file(output_seq, fallback)
        write_report(
            output_seq.with_suffix(output_seq.suffix + ".json"),
            {
                "input_aig": str(input_aig),
                "output_seq": str(output_seq),
                "evaluated_candidates": 0,
                "max_steps": args.max_steps,
                "use_history": use_history,
                "fallback": fallback,
                "elapsed_sec": time.time() - start,
            },
        )
        return 0

    best_commands = build_sequence(best.candidate, use_history=use_history)
    best_sequence = serialize_sequence(best_commands)
    write_sequence_file(output_seq, best_sequence)
    write_report(
        output_seq.with_suffix(output_seq.suffix + ".json"),
        {
            "input_aig": str(input_aig),
            "output_seq": str(output_seq),
            "evaluated_candidates": evaluated,
            "max_steps": args.max_steps,
            "use_history": use_history,
            "best_cost": best.cost,
            "best_area": best.area,
            "best_depth": best.depth,
            "best_flow_labels": candidate_to_labels(best.candidate),
            "best_sequence": best_sequence,
            "elapsed_sec": time.time() - start,
        },
    )

    if not args.quiet:
        print(
            f"done: cost={best.cost:.3f} depth={best.depth} area={best.area} "
            f"evaluated={evaluated}/{total}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
