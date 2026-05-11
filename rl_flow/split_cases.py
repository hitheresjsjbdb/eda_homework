#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Split public cases into train/eval/test.")
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--eval", dest="eval_size", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    args = parser.parse_args()

    case_dirs = sorted([p for p in args.case_root.iterdir() if p.is_dir()])
    case_names = [p.name for p in case_dirs]
    total = len(case_names)
    if args.train + args.eval_size + args.test > total:
        raise SystemExit("requested split sizes exceed case count")

    rng = random.Random(args.seed)
    rng.shuffle(case_names)

    train = case_names[: args.train]
    eval_cases = case_names[args.train : args.train + args.eval_size]
    test = case_names[args.train + args.eval_size : args.train + args.eval_size + args.test]

    output = {
        "train": train,
        "eval": eval_cases,
        "test": test,
        "meta": {
            "seed": args.seed,
            "total_cases": total,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
