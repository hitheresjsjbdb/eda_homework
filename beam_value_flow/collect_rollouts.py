#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from common import (
    DEFAULT_ACTIONS,
    DEFAULT_MAP_ACTIONS,
    IMAP_BIN,
    aig_sha1,
    discover_aigs,
)

import sys

if str((Path(__file__).resolve().parents[1] / "iMAP" / "eda23")) not in sys.path:
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "iMAP" / "eda23")))

from bruteforce_search import Candidate, candidate_to_labels, evaluate_candidate  # noqa: E402
from generate_dataset import hybrid_candidates, random_candidates  # noqa: E402


def iter_candidates(strategy: str, max_steps: int, exhaustive_depth: int, random_samples: int, rng: random.Random):
    if strategy == "random":
        yield from random_candidates(
            DEFAULT_ACTIONS,
            DEFAULT_MAP_ACTIONS,
            max_steps=max_steps,
            num_samples=random_samples,
            rng=rng,
        )
        return
    if strategy == "hybrid":
        yield from hybrid_candidates(
            DEFAULT_ACTIONS,
            DEFAULT_MAP_ACTIONS,
            exhaustive_depth=exhaustive_depth,
            random_max_steps=max_steps,
            random_samples=random_samples,
            rng=rng,
        )
        return
    raise ValueError(f"unsupported strategy: {strategy}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect rollout data for value-model training.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--imap-bin", type=Path, default=IMAP_BIN)
    parser.add_argument("--strategy", choices=("random", "hybrid"), default="hybrid")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--exhaustive-depth", type=int, default=1)
    parser.add_argument("--random-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    aig_files = discover_aigs(args.input_root)
    if args.limit_cases is not None:
        aig_files = aig_files[: args.limit_cases]
    if not aig_files:
        raise SystemExit("no AIG files found")

    output_jsonl = args.output_jsonl.resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    use_history = not args.no_history
    total = 0
    started = time.time()

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for case_index, aig_path in enumerate(aig_files, start=1):
            case_rng = random.Random(rng.randint(0, 2**31 - 1))
            best = None
            print(f"[{case_index}/{len(aig_files)}] {aig_path.stem}", flush=True)
            for sample_index, candidate in enumerate(
                iter_candidates(
                    strategy=args.strategy,
                    max_steps=args.max_steps,
                    exhaustive_depth=args.exhaustive_depth,
                    random_samples=args.random_samples,
                    rng=case_rng,
                ),
                start=1,
            ):
                result = evaluate_candidate(
                    aig_path=aig_path,
                    imap_bin=args.imap_bin.resolve(),
                    candidate=candidate,
                    use_history=use_history,
                    timeout=args.timeout,
                )
                if result is None:
                    continue
                total += 1
                record = {
                    "case_name": aig_path.stem,
                    "case_path": str(aig_path.resolve()),
                    "case_sha1": aig_sha1(aig_path),
                    "sample_index": sample_index,
                    "step_labels": [step.name for step in candidate.steps],
                    "map_label": candidate.map_action.name,
                    "flow_labels": candidate_to_labels(candidate),
                    "step_count": len(candidate.steps),
                    "area": result.area,
                    "depth": result.depth,
                    "cost": result.cost,
                    "use_history": use_history,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                if best is None or (result.cost, result.depth, result.area) < (best.cost, best.depth, best.area):
                    best = result
                    print(
                        f"  best cost={result.cost:.3f} area={result.area} depth={result.depth} "
                        f"steps={[step.name for step in candidate.steps]} map={candidate.map_action.name}",
                        flush=True,
                    )

    summary = {
        "input_root": str(args.input_root.resolve()),
        "output_jsonl": str(output_jsonl),
        "cases": len(aig_files),
        "records": total,
        "elapsed_sec": time.time() - started,
        "strategy": args.strategy,
        "max_steps": args.max_steps,
        "exhaustive_depth": args.exhaustive_depth,
        "random_samples": args.random_samples,
        "use_history": use_history,
    }
    output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
