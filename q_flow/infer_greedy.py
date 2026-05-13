#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_flow.actions import default_actions
from q_flow.common import resolve_device
from q_flow.env import AIGEnv
from q_flow.inference import load_checkpoint, run_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer a sequence with fast greedy Q policy.")
    parser.add_argument("input_aig", type=Path)
    parser.add_argument("output_seq", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imap-bin", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--confidence-margin", type=float, default=0.08)
    parser.add_argument("--fallback-topk", type=int, default=3)
    parser.add_argument("--fallback-depth", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    actions = default_actions()
    env = AIGEnv(
        input_aig=args.input_aig,
        imap_bin=args.imap_bin,
        actions=actions,
        max_steps=args.max_steps,
        timeout_sec=args.timeout_sec,
    )
    model = load_checkpoint(args.checkpoint, device)
    result = run_policy(
        env=env,
        actions=actions,
        model=model,
        device=device,
        confidence_margin=args.confidence_margin,
        fallback_topk=args.fallback_topk,
        fallback_depth=args.fallback_depth,
    )

    args.output_seq.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq.write_text("; ".join(result.command_sequence) + ";\n", encoding="utf-8")
    print(
        f"cost={result.final_stats.cost:.4f} area={result.final_stats.area} "
        f"depth={result.final_stats.depth} actions={result.action_indices}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
