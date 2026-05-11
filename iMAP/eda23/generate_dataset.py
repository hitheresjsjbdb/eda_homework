#!/usr/bin/env python3
import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

from bruteforce_search import (
    DEFAULT_ACTIONS,
    DEFAULT_MAP_ACTIONS,
    Candidate,
    build_sequence,
    candidate_count,
    candidate_to_labels,
    evaluate_candidate,
    generate_candidates,
    serialize_sequence,
)


def discover_aigs(root: Path) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        if root.suffix != ".aig":
            raise SystemExit(f"expected .aig file, got: {root}")
        return [root]
    if not root.is_dir():
        raise SystemExit(f"path not found: {root}")
    return sorted(root.rglob("*.aig"))


def random_candidates(
    actions,
    map_actions,
    max_steps: int,
    num_samples: int,
    rng: random.Random,
):
    seen: set[tuple[str, ...]] = set()
    while len(seen) < num_samples:
        depth = rng.randint(0, max_steps)
        steps = tuple(rng.choice(actions) for _ in range(depth))
        map_action = rng.choice(map_actions)
        key = tuple(step.name for step in steps) + (map_action.name,)
        if key in seen:
            continue
        seen.add(key)
        yield Candidate(steps=steps, map_action=map_action)


def hybrid_candidates(
    actions,
    map_actions,
    exhaustive_depth: int,
    random_max_steps: int,
    random_samples: int,
    rng: random.Random,
):
    seen: set[tuple[str, ...]] = set()
    for candidate in generate_candidates(actions, map_actions, exhaustive_depth):
        key = tuple(step.name for step in candidate.steps) + (candidate.map_action.name,)
        seen.add(key)
        yield candidate

    for candidate in random_candidates(
        actions, map_actions, random_max_steps, random_samples, rng
    ):
        key = tuple(step.name for step in candidate.steps) + (candidate.map_action.name,)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def iter_candidates(
    strategy: str,
    exhaustive_depth: int,
    max_steps: int,
    random_samples_count: int,
    rng: random.Random,
):
    if strategy == "exhaustive":
        yield from generate_candidates(DEFAULT_ACTIONS, DEFAULT_MAP_ACTIONS, max_steps)
        return
    if strategy == "random":
        yield from random_candidates(
            DEFAULT_ACTIONS,
            DEFAULT_MAP_ACTIONS,
            max_steps=max_steps,
            num_samples=random_samples_count,
            rng=rng,
        )
        return
    if strategy == "hybrid":
        yield from hybrid_candidates(
            DEFAULT_ACTIONS,
            DEFAULT_MAP_ACTIONS,
            exhaustive_depth=exhaustive_depth,
            random_max_steps=max_steps,
            random_samples=random_samples_count,
            rng=rng,
        )
        return
    raise ValueError(f"unsupported strategy: {strategy}")


def sequence_without_io(candidate: Candidate, use_history: bool) -> str:
    return serialize_sequence(build_sequence(candidate, use_history=use_history))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate training data for EDA23 by evaluating many iMAP flows."
    )
    parser.add_argument("input_root", type=Path, help="AIG file or directory of AIGs")
    parser.add_argument("output_jsonl", type=Path, help="Output dataset in JSONL format")
    parser.add_argument(
        "--imap-bin",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bin" / "imap",
    )
    parser.add_argument(
        "--strategy",
        choices=("exhaustive", "random", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum optimization sequence length.",
    )
    parser.add_argument(
        "--exhaustive-depth",
        type=int,
        default=2,
        help="For hybrid mode, enumerate all sequences up to this depth.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=400,
        help="For random/hybrid mode, number of random candidates per case.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional per-candidate timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Only process the first N discovered AIGs.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history-based mapping.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    args = parser.parse_args()

    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    if args.exhaustive_depth < 0:
        raise SystemExit("--exhaustive-depth must be non-negative")
    if args.exhaustive_depth > args.max_steps:
        raise SystemExit("--exhaustive-depth cannot exceed --max-steps")

    imap_bin = args.imap_bin.resolve()
    if not imap_bin.is_file():
        raise SystemExit(f"imap binary not found: {imap_bin}")

    aig_files = discover_aigs(args.input_root)
    if args.limit_cases is not None:
        aig_files = aig_files[: args.limit_cases]
    if not aig_files:
        raise SystemExit("no AIG files found")

    use_history = not args.no_history
    output_jsonl = args.output_jsonl.resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")

    global_rng = random.Random(args.seed)

    if not args.quiet:
        if args.strategy == "exhaustive":
            estimate = candidate_count(
                len(DEFAULT_ACTIONS), len(DEFAULT_MAP_ACTIONS), args.max_steps
            )
        elif args.strategy == "random":
            estimate = args.random_samples
        else:
            estimate = candidate_count(
                len(DEFAULT_ACTIONS), len(DEFAULT_MAP_ACTIONS), args.exhaustive_depth
            ) + args.random_samples
        print(
            f"dataset generation: cases={len(aig_files)} strategy={args.strategy} "
            f"approx_candidates_per_case={estimate} use_history={use_history}",
            file=sys.stderr,
        )

    total_records = 0
    global_best = None
    case_summaries = []
    started = time.time()

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for case_index, aig_path in enumerate(aig_files):
            case_rng = random.Random(global_rng.randint(0, 2**31 - 1))
            best_record = None
            evaluated = 0
            case_started = time.time()

            if not args.quiet:
                print(
                    f"[{case_index + 1}/{len(aig_files)}] {aig_path.name}",
                    file=sys.stderr,
                )

            for sample_index, candidate in enumerate(
                iter_candidates(
                    strategy=args.strategy,
                    exhaustive_depth=args.exhaustive_depth,
                    max_steps=args.max_steps,
                    random_samples_count=args.random_samples,
                    rng=case_rng,
                ),
                start=1,
            ):
                eval_started = time.time()
                result = evaluate_candidate(
                    aig_path=aig_path,
                    imap_bin=imap_bin,
                    candidate=candidate,
                    use_history=use_history,
                    timeout=args.timeout,
                )
                if result is None:
                    continue

                evaluated += 1
                total_records += 1
                record = {
                    "case_name": aig_path.stem,
                    "case_path": str(aig_path),
                    "strategy": args.strategy,
                    "sample_index": sample_index,
                    "step_count": len(candidate.steps),
                    "flow_labels": candidate_to_labels(candidate),
                    "sequence": sequence_without_io(candidate, use_history=use_history),
                    "area": result.area,
                    "depth": result.depth,
                    "cost": result.cost,
                    "eval_time_sec": time.time() - eval_started,
                    "use_history": use_history,
                }
                out_f.write(json.dumps(record, sort_keys=True) + "\n")

                if (
                    best_record is None
                    or record["cost"] < best_record["cost"]
                    or (
                        record["cost"] == best_record["cost"]
                        and (record["depth"], record["area"])
                        < (best_record["depth"], best_record["area"])
                    )
                ):
                    best_record = record
                    if not args.quiet:
                        print(
                            f"  best cost={record['cost']:.3f} depth={record['depth']} "
                            f"area={record['area']} labels={record['flow_labels']}",
                            file=sys.stderr,
                        )

            case_summary = {
                "case_name": aig_path.stem,
                "case_path": str(aig_path),
                "evaluated_records": evaluated,
                "elapsed_sec": time.time() - case_started,
                "best": best_record,
            }
            case_summaries.append(case_summary)

            if best_record is not None and (
                global_best is None or best_record["cost"] < global_best["cost"]
            ):
                global_best = best_record

    summary = {
        "input_root": str(args.input_root.resolve()),
        "output_jsonl": str(output_jsonl),
        "strategy": args.strategy,
        "max_steps": args.max_steps,
        "exhaustive_depth": args.exhaustive_depth,
        "random_samples": args.random_samples,
        "use_history": use_history,
        "seed": args.seed,
        "case_count": len(aig_files),
        "total_records": total_records,
        "elapsed_sec": time.time() - started,
        "global_best": global_best,
        "cases": case_summaries,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not args.quiet:
        print(
            f"done: records={total_records} cases={len(aig_files)} "
            f"summary={summary_path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
