#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_flow.actions import default_macro_actions
from rl_flow.budget import adapt_search_budget
from rl_flow.imap_env import ImapEnv
from rl_flow.policy_search import beam_search, load_policy_checkpoint
from rl_flow.utils import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer a .seq from a trained policy.")
    parser.add_argument("input_aig", type=Path)
    parser.add_argument("output_seq", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imap-bin", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--beam-branch-topk", type=int, default=4)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    actions = default_macro_actions()
    model, _checkpoint = load_policy_checkpoint(args.checkpoint, device)

    print(f"infer device: {device}")

    env = ImapEnv(
        input_aig=args.input_aig,
        imap_bin=args.imap_bin,
        actions=actions,
        max_steps=args.max_steps,
        timeout_sec=args.timeout_sec,
    )
    env.reset()
    if env.initial_snapshot is None:
        raise RuntimeError("failed to initialize environment")
    init_cost = float(env.initial_snapshot.cost)
    max_steps, beam_width, branch_topk = adapt_search_budget(
        init_cost,
        args.max_steps,
        args.beam_width,
        args.beam_branch_topk,
    )
    env.max_steps = max_steps

    result = beam_search(
        env=env,
        model=model,
        device=device,
        beam_width=beam_width,
        branch_topk=branch_topk,
        temperature=args.temperature,
        expand_workers=args.search_workers,
        reset_env=False,
    )

    args.output_seq.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq.write_text(result["sequence"] + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
