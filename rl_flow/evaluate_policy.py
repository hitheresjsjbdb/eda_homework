#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_flow.actions import default_macro_actions
from rl_flow.budget import adapt_search_budget
from rl_flow.imap_env import ImapEnv
from rl_flow.policy_search import beam_search, load_policy_checkpoint
from rl_flow.progress import progress_iter
from rl_flow.utils import load_split, read_ref_qor, resolve_device


_EVAL_MODEL = None
_EVAL_ACTIONS = None
_EVAL_CASE_ROOT: Path | None = None
_EVAL_IMAP_BIN: Path | None = None
_EVAL_MAX_STEPS = 4
_EVAL_TIMEOUT_SEC = 60.0
_EVAL_BEAM_WIDTH = 5
_EVAL_BRANCH_TOPK = 4
_EVAL_TEMPERATURE = 1.0
_EVAL_EXPAND_WORKERS = 1


def _init_eval_worker(
    checkpoint_path: str,
    case_root: str,
    imap_bin: str,
    max_steps: int,
    timeout_sec: float,
    beam_width: int,
    branch_topk: int,
    temperature: float,
    expand_workers: int,
) -> None:
    global _EVAL_MODEL
    global _EVAL_ACTIONS
    global _EVAL_CASE_ROOT
    global _EVAL_IMAP_BIN
    global _EVAL_MAX_STEPS
    global _EVAL_TIMEOUT_SEC
    global _EVAL_BEAM_WIDTH
    global _EVAL_BRANCH_TOPK
    global _EVAL_TEMPERATURE
    global _EVAL_EXPAND_WORKERS

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    _EVAL_ACTIONS = default_macro_actions()
    _EVAL_MODEL, _ = load_policy_checkpoint(Path(checkpoint_path), resolve_device("cpu"))
    _EVAL_CASE_ROOT = Path(case_root)
    _EVAL_IMAP_BIN = Path(imap_bin)
    _EVAL_MAX_STEPS = max_steps
    _EVAL_TIMEOUT_SEC = timeout_sec
    _EVAL_BEAM_WIDTH = beam_width
    _EVAL_BRANCH_TOPK = branch_topk
    _EVAL_TEMPERATURE = temperature
    _EVAL_EXPAND_WORKERS = expand_workers


def _evaluate_case_worker(case_name: str) -> dict[str, object]:
    if _EVAL_MODEL is None or _EVAL_CASE_ROOT is None or _EVAL_IMAP_BIN is None:
        raise RuntimeError("eval worker not initialized")

    case_dir = _EVAL_CASE_ROOT / case_name
    aig_path = case_dir / f"{case_name}.aig"
    env = ImapEnv(
        input_aig=aig_path,
        imap_bin=_EVAL_IMAP_BIN,
        actions=_EVAL_ACTIONS,
        max_steps=_EVAL_MAX_STEPS,
        timeout_sec=_EVAL_TIMEOUT_SEC,
    )
    env.reset()
    if env.initial_snapshot is None:
        raise RuntimeError("failed to initialize environment")
    init_cost = float(env.initial_snapshot.cost)
    max_steps, beam_width, branch_topk = adapt_search_budget(
        init_cost,
        _EVAL_MAX_STEPS,
        _EVAL_BEAM_WIDTH,
        _EVAL_BRANCH_TOPK,
    )
    env.max_steps = max_steps
    final_info = beam_search(
        env=env,
        model=_EVAL_MODEL,
        device=resolve_device("cpu"),
        beam_width=beam_width,
        branch_topk=branch_topk,
        temperature=_EVAL_TEMPERATURE,
        expand_workers=_EVAL_EXPAND_WORKERS,
        reset_env=False,
    )
    ref = read_ref_qor(case_dir)
    item = {
        "case_name": case_name,
        "cost": float(final_info["cost"]),
        "area": int(final_info["area"]),
        "depth": int(final_info["depth"]),
        "sequence": str(final_info["sequence"]),
        "ref": ref,
    }
    if ref is not None:
        item["ref_area_gap"] = int(final_info["area"]) - ref["area"]
        item["ref_depth_gap"] = int(final_info["depth"]) - ref["level"]
        item["ref_cost_gap"] = float(final_info["cost"]) - (0.6 * ref["level"] + 0.4 * ref["area"])
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy on a case split.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "eval", "test"), required=True)
    parser.add_argument("--case-root", type=Path, default=Path("/home/pan/eda/23_question2/iMAP/eda23/benchmark_public"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imap-bin", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--beam-branch-topk", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    case_names = load_split(args.split, args.split_name)
    actions = default_macro_actions()
    device = resolve_device(args.device)
    model, _checkpoint = load_policy_checkpoint(args.checkpoint, device)
    num_workers = args.num_workers if args.num_workers > 0 else min(8, os.cpu_count() or 1)

    print(f"eval device: {device}")
    print(f"eval workers: {num_workers}")

    results = []
    total_cost = 0.0
    exact_ref_matches = 0
    ref_gap_sum = 0.0
    ref_area_gap_sum = 0.0
    ref_depth_gap_sum = 0.0
    ref_count = 0
    infer_better_count = 0
    ref_better_count = 0
    tie_count = 0

    if num_workers > 1 and len(case_names) > 1:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_eval_worker,
            initargs=(
                str(args.checkpoint),
                str(args.case_root),
                str(args.imap_bin),
                args.max_steps,
                args.timeout_sec,
                args.beam_width,
                args.beam_branch_topk,
                args.temperature,
                args.search_workers,
            ),
        ) as executor:
            futures = [executor.submit(_evaluate_case_worker, case_name) for case_name in case_names]
            for future in progress_iter(as_completed(futures), total=len(futures), desc="eval", unit="case"):
                try:
                    item = future.result()
                except Exception as exc:
                    print(f"eval worker failed: {exc}")
                    continue
                ref = item["ref"]
                if ref is not None:
                    ref_cost = 0.6 * ref["level"] + 0.4 * ref["area"]
                    ref_count += 1
                    ref_gap_sum += float(item["ref_cost_gap"])
                    ref_area_gap_sum += float(item["ref_area_gap"])
                    ref_depth_gap_sum += float(item["ref_depth_gap"])
                    if float(item["cost"]) < ref_cost:
                        infer_better_count += 1
                    elif float(item["cost"]) > ref_cost:
                        ref_better_count += 1
                    else:
                        tie_count += 1
                    if int(item["area"]) == ref["area"] and int(item["depth"]) == ref["level"]:
                        exact_ref_matches += 1
                total_cost += float(item["cost"])
                results.append(item)
    else:
        for case_name in progress_iter(case_names, desc="eval", unit="case"):
            case_dir = args.case_root / case_name
            aig_path = case_dir / f"{case_name}.aig"
            env = ImapEnv(
                input_aig=aig_path,
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
            final_info = beam_search(
                env=env,
                model=model,
                device=device,
                beam_width=beam_width,
                branch_topk=branch_topk,
                temperature=args.temperature,
                expand_workers=args.search_workers,
                reset_env=False,
            )

            ref = read_ref_qor(case_dir)
            item = {
                "case_name": case_name,
                "cost": float(final_info["cost"]),
                "area": int(final_info["area"]),
                "depth": int(final_info["depth"]),
                "sequence": str(final_info["sequence"]),
                "ref": ref,
            }
            if ref is not None:
                item["ref_area_gap"] = int(final_info["area"]) - ref["area"]
                item["ref_depth_gap"] = int(final_info["depth"]) - ref["level"]
                ref_cost = 0.6 * ref["level"] + 0.4 * ref["area"]
                item["ref_cost_gap"] = float(final_info["cost"]) - ref_cost
                ref_count += 1
                ref_gap_sum += float(item["ref_cost_gap"])
                ref_area_gap_sum += float(item["ref_area_gap"])
                ref_depth_gap_sum += float(item["ref_depth_gap"])
                if float(item["cost"]) < ref_cost:
                    infer_better_count += 1
                elif float(item["cost"]) > ref_cost:
                    ref_better_count += 1
                else:
                    tie_count += 1
                if int(final_info["area"]) == ref["area"] and int(final_info["depth"]) == ref["level"]:
                    exact_ref_matches += 1
            total_cost += float(final_info["cost"])
            results.append(item)

    summary = {
        "split_name": args.split_name,
        "case_count": len(results),
        "avg_cost": total_cost / max(1, len(results)),
        "avg_ref_gap": ref_gap_sum / max(1, ref_count),
        "avg_ref_area_gap": ref_area_gap_sum / max(1, ref_count),
        "avg_ref_depth_gap": ref_depth_gap_sum / max(1, ref_count),
        "ref_count": ref_count,
        "exact_ref_matches": exact_ref_matches,
        "infer_better_count": infer_better_count,
        "ref_better_count": ref_better_count,
        "tie_count": tie_count,
        "infer_better_pct": 100.0 * infer_better_count / max(1, ref_count),
        "ref_better_pct": 100.0 * ref_better_count / max(1, ref_count),
        "tie_pct": 100.0 * tie_count / max(1, ref_count),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    comparable_results = [item for item in results if item["ref"] is not None]
    comparable_results.sort(key=lambda item: item["case_name"])
    for item in comparable_results:
        ref = item["ref"]
        ref_cost = 0.6 * ref["level"] + 0.4 * ref["area"]
        infer_cost = float(item["cost"])
        gap = abs(infer_cost - ref_cost)
        if infer_cost < ref_cost:
            winner = "infer_better"
        elif infer_cost > ref_cost:
            winner = "ref_better"
        else:
            winner = "tie"
        print(
            f"{item['case_name']}: "
            f"ref_cost={ref_cost:.4f} "
            f"infer_cost={infer_cost:.4f} "
            f"winner={winner} "
            f"gap={gap:.4f}"
        )

    print(
        f"cases={len(results)} "
        f"avg_cost={summary['avg_cost']:.3f} "
        f"avg_ref_gap={summary['avg_ref_gap']:.3f} "
        f"exact_ref_matches={exact_ref_matches} "
        f"infer_better={infer_better_count}/{max(1, ref_count)} ({summary['infer_better_pct']:.2f}%) "
        f"ref_better={ref_better_count}/{max(1, ref_count)} ({summary['ref_better_pct']:.2f}%) "
        f"tie={tie_count}/{max(1, ref_count)} ({summary['tie_pct']:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
