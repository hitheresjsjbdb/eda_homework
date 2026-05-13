#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_flow.actions import default_actions
from q_flow.common import load_split, progress_iter, read_ref_qor, resolve_device
from q_flow.env import AIGEnv
from q_flow.inference import load_checkpoint, run_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fast Q policy on a split.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "eval", "test"), required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imap-bin", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--confidence-margin", type=float, default=0.08)
    parser.add_argument("--fallback-topk", type=int, default=3)
    parser.add_argument("--fallback-depth", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    actions = default_actions()
    model = load_checkpoint(args.checkpoint, device)
    case_names = load_split(args.split, args.split_name)

    results = []
    total_cost = 0.0
    fail_count = 0
    for case_name in progress_iter(case_names, desc="eval", unit="case"):
        case_dir = args.case_root / case_name
        env = AIGEnv(
            input_aig=case_dir / f"{case_name}.aig",
            imap_bin=args.imap_bin,
            actions=actions,
            max_steps=args.max_steps,
            timeout_sec=args.timeout_sec,
        )
        try:
            result = run_policy(
                env=env,
                actions=actions,
                model=model,
                device=device,
                confidence_margin=args.confidence_margin,
                fallback_topk=args.fallback_topk,
                fallback_depth=args.fallback_depth,
            )
        except Exception as exc:
            fail_count += 1
            results.append({"case_name": case_name, "error": str(exc)})
            continue

        item = {
            "case_name": case_name,
            "cost": float(result.final_stats.cost),
            "area": int(result.final_stats.area),
            "depth": int(result.final_stats.depth),
            "actions": result.action_indices,
            "sequence": result.command_sequence,
        }
        ref = read_ref_qor(case_dir)
        if ref is not None:
            ref_cost = 0.6 * ref["level"] + 0.4 * ref["area"]
            item["ref_cost_gap"] = item["cost"] - ref_cost
        total_cost += item["cost"]
        results.append(item)

    success_count = max(0, len(results) - fail_count)
    output = {
        "case_count": len(results),
        "success_count": success_count,
        "fail_count": fail_count,
        "avg_cost": total_cost / max(1, success_count),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"avg_cost={output['avg_cost']:.4f} success={success_count}/{len(results)} "
        f"fails={fail_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
