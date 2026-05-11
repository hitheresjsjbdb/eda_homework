#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import IMAP_BIN, discover_aigs
from search import search_case


def parse_ref_qor(case_dir: Path) -> tuple[int, int] | None:
    ref_path = case_dir / "ref_qor.txt"
    if not ref_path.is_file():
        return None
    area = None
    depth = None
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("area:"):
            area = int(line.split(":", 1)[1].strip())
        elif line.startswith("level:"):
            depth = int(line.split(":", 1)[1].strip())
    if area is None or depth is None:
        return None
    return area, depth


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate beam-value flow on public cases.")
    parser.add_argument("public_root", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--imap-bin", type=Path, default=IMAP_BIN)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--top-actions", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.public_root.resolve().iterdir() if path.is_dir())
    if args.limit_cases is not None:
        case_dirs = case_dirs[: args.limit_cases]
    results = []
    ref_matches = 0
    exact_matches = 0
    total_cost_gap = 0.0
    total_ratio = 0.0

    for index, case_dir in enumerate(case_dirs, start=1):
        aig_path = case_dir / f"{case_dir.name}.aig"
        if not aig_path.is_file():
            continue
        print(f"[{index}/{len(case_dirs)}] {case_dir.name}", flush=True)
        result = search_case(
            aig_path=aig_path,
            model_path=args.model_path.resolve(),
            imap_bin=args.imap_bin.resolve(),
            beam_width=args.beam_width,
            top_actions=args.top_actions,
            max_steps=args.max_steps,
            timeout=args.timeout,
        )
        ref = parse_ref_qor(case_dir)
        if ref is not None:
            ref_area, ref_depth = ref
            ref_cost = 0.4 * ref_area + 0.6 * ref_depth
            result["ref_area"] = ref_area
            result["ref_depth"] = ref_depth
            result["ref_cost"] = ref_cost
            result["cost_gap"] = result["best_cost"] - ref_cost
            result["cost_ratio"] = result["best_cost"] / ref_cost
            ref_matches += 1
            total_cost_gap += result["cost_gap"]
            total_ratio += result["cost_ratio"]
            if result["best_area"] == ref_area and result["best_depth"] == ref_depth:
                exact_matches += 1
        results.append(result)

    summary = {
        "cases": len(results),
        "cases_with_ref": ref_matches,
        "exact_matches": exact_matches,
        "avg_cost_gap": (total_cost_gap / ref_matches) if ref_matches else None,
        "avg_cost_ratio": (total_ratio / ref_matches) if ref_matches else None,
        "beam_width": args.beam_width,
        "top_actions": args.top_actions,
        "max_steps": args.max_steps,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
