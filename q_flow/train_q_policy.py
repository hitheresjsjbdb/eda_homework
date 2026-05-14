#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import random
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_flow.actions import default_actions
from q_flow.common import load_split, resolve_device
from q_flow.env import AIGEnv
from q_flow.model import GreedyQNet
from q_flow.teacher import TeacherRecord, build_teacher_records, estimate_teacher_states

_CACHE_VERSION = 2


def minibatch_indices(size: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    indices = list(range(size))
    rng.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, size, batch_size)]


def _cache_key(
    case_name: str,
    max_steps: int,
    branch_topk: int,
    deep_branch_topk: int,
    tail_branch_topk: int,
    terminal_topk: int,
    deep_terminal_topk: int,
    timeout_sec: float,
    teacher_budget_sec: float,
    teacher_aig_timeout_sec: float,
    teacher_final_timeout_sec: float,
    action_names: list[str],
) -> str:
    payload = json.dumps(
        {
            "cache_version": _CACHE_VERSION,
            "case_name": case_name,
            "max_steps": max_steps,
            "branch_topk": branch_topk,
            "deep_branch_topk": deep_branch_topk,
            "tail_branch_topk": tail_branch_topk,
            "terminal_topk": terminal_topk,
            "deep_terminal_topk": deep_terminal_topk,
            "timeout_sec": timeout_sec,
            "teacher_budget_sec": teacher_budget_sec,
            "teacher_aig_timeout_sec": teacher_aig_timeout_sec,
            "teacher_final_timeout_sec": teacher_final_timeout_sec,
            "action_names": action_names,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _load_cached_records(cache_file: Path) -> list[TeacherRecord] | None:
    if not cache_file.is_file():
        return None
    data = np.load(cache_file, allow_pickle=False)
    count = int(data["count"][0])
    records: list[TeacherRecord] = []
    for idx in range(count):
        payload = {
            "obs": data[f"obs_{idx}"],
            "best_action": int(data[f"best_action_{idx}"][0]),
            "q_values": data[f"q_values_{idx}"],
            "action_mask": data[f"action_mask_{idx}"],
            "target_value": float(data[f"target_value_{idx}"][0]),
            "target_area": float(data[f"target_area_{idx}"][0]),
            "target_depth": float(data[f"target_depth_{idx}"][0]),
        }
        records.append(TeacherRecord.from_payload(payload))
    return records


def _is_hex_hash(text: str) -> bool:
    if len(text) != 16:
        return False
    return all(ch in "0123456789abcdef" for ch in text)


def _find_cache_file(cache_dir: Path, case_name: str, cache_key: str) -> tuple[Path | None, str]:
    exact = cache_dir / f"{case_name}_{cache_key}.npz"
    if exact.is_file():
        return exact, "exact"

    legacy_candidates: list[Path] = []
    for candidate in cache_dir.glob(f"{case_name}_*.npz"):
        stem = candidate.stem
        if "_" not in stem:
            continue
        prefix, suffix = stem.rsplit("_", 1)
        if prefix != case_name:
            continue
        if not _is_hex_hash(suffix):
            continue
        legacy_candidates.append(candidate)

    if not legacy_candidates:
        return None, "missing"
    legacy_candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return legacy_candidates[0], "legacy"


def _save_cached_records(cache_file: Path, records: list[TeacherRecord]) -> None:
    arrays: dict[str, np.ndarray] = {"count": np.asarray([len(records)], dtype=np.int32)}
    for idx, record in enumerate(records):
        payload = record.to_payload()
        arrays[f"obs_{idx}"] = np.asarray(payload["obs"], dtype=np.float32)
        arrays[f"best_action_{idx}"] = np.asarray([payload["best_action"]], dtype=np.int32)
        arrays[f"q_values_{idx}"] = np.asarray(payload["q_values"], dtype=np.float32)
        arrays[f"action_mask_{idx}"] = np.asarray(payload["action_mask"], dtype=np.float32)
        arrays[f"target_value_{idx}"] = np.asarray([payload["target_value"]], dtype=np.float32)
        arrays[f"target_area_{idx}"] = np.asarray([payload["target_area"]], dtype=np.float32)
        arrays[f"target_depth_{idx}"] = np.asarray([payload["target_depth"]], dtype=np.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **arrays)


def masked_q_loss(pred_q: torch.Tensor, target_q: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    per_item = nn.functional.smooth_l1_loss(pred_q, target_q, reduction="none")
    masked = per_item * action_mask
    denom = action_mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def _emit_event(event_queue, event: tuple[object, ...]) -> None:
    if event_queue is None:
        return
    try:
        event_queue.put(event)
    except Exception:
        return


def _collect_case_records(task: dict[str, object]) -> dict[str, object]:
    case_name = str(task["case_name"])
    cache_file = Path(str(task["cache_file"]))
    cache_dir = cache_file.parent
    cache_key = str(task["cache_key"])
    event_queue = task.get("event_queue")
    if not bool(task["rebuild_cache"]):
        cache_path, cache_mode = _find_cache_file(cache_dir, case_name, cache_key)
        cached = _load_cached_records(cache_path) if cache_path is not None else None
        if cached is not None:
            if cache_mode == "legacy" and cache_path is not None and cache_path != cache_file:
                try:
                    shutil.copy2(cache_path, cache_file)
                except Exception:
                    pass
            _emit_event(event_queue, ("case_done", case_name, len(cached), f"cache-{cache_mode}"))
            return {
                "case_name": case_name,
                "cache_file": str(cache_file if cache_mode == "legacy" else cache_path),
                "record_count": len(cached),
                "source": f"cache-{cache_mode}",
            }

    actions = default_actions()
    env = AIGEnv(
        input_aig=Path(str(task["input_aig"])),
        imap_bin=Path(str(task["imap_bin"])),
        actions=actions,
        max_steps=int(task["max_steps"]),
        timeout_sec=float(task["teacher_aig_timeout_sec"]),
        final_timeout_sec=float(task["teacher_final_timeout_sec"]),
    )
    _emit_event(event_queue, ("case_start", case_name, int(task["estimated_states"])))

    def on_progress(delta: int) -> None:
        _emit_event(event_queue, ("case_progress", case_name, delta))

    try:
        records = build_teacher_records(
            env=env,
            actions=actions,
            max_steps=int(task["max_steps"]),
            branch_topk=int(task["branch_topk"]),
            deep_branch_topk=int(task["deep_branch_topk"]),
            tail_branch_topk=int(task["tail_branch_topk"]),
            terminal_topk=int(task["terminal_topk"]),
            deep_terminal_topk=int(task["deep_terminal_topk"]),
            teacher_budget_sec=float(task["teacher_budget_sec"]),
            progress_callback=on_progress,
        )
        _save_cached_records(cache_file, records)
        _emit_event(event_queue, ("case_done", case_name, len(records), "build"))
        return {
            "case_name": case_name,
            "cache_file": str(cache_file),
            "record_count": len(records),
            "source": "build",
        }
    except Exception as exc:
        _emit_event(event_queue, ("case_error", case_name, str(exc)))
        return {"case_name": case_name, "record_count": 0, "source": "error", "error": str(exc)}


def _progress_monitor(event_queue, total_cases: int) -> None:
    try:
        from tqdm import tqdm
    except Exception:
        while True:
            event = event_queue.get()
            if event[0] == "all_done":
                return
        return

    overall = tqdm(total=total_cases, desc="teacher", unit="case", position=0)
    bars: dict[str, object] = {}
    positions: dict[str, int] = {}
    next_position = 1
    completed = 0
    while True:
        try:
            event = event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        except (EOFError, BrokenPipeError, OSError):
            break
        kind = event[0]
        if kind == "all_done":
            break
        if kind == "case_start":
            _kind, case_name, total = event
            if case_name not in bars:
                positions[case_name] = next_position
                next_position += 1
                bars[case_name] = tqdm(
                    total=max(1, int(total)),
                    desc=case_name,
                    unit="state",
                    position=positions[case_name],
                    leave=False,
                )
        elif kind == "case_progress":
            _kind, case_name, delta = event
            bar = bars.get(case_name)
            if bar is not None:
                bar.update(int(delta))
        elif kind == "case_done":
            _kind, case_name, record_count, source = event
            bar = bars.get(case_name)
            if bar is not None:
                if getattr(bar, "total", None) is not None and bar.n < bar.total:
                    bar.update(bar.total - bar.n)
                bar.set_postfix_str(f"{source},{record_count} rec")
                bar.close()
                del bars[case_name]
            completed += 1
            overall.update(1)
        elif kind == "case_error":
            _kind, case_name, message = event
            bar = bars.get(case_name)
            if bar is not None:
                bar.set_postfix_str("error")
                bar.close()
                del bars[case_name]
            completed += 1
            overall.update(1)
            overall.write(f"[teacher-error] {case_name}: {message}")
    for bar in bars.values():
        bar.close()
    overall.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an offline greedy Q policy for EDA23.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--imap-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("./tmp/q_flow_cache"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--branch-topk", type=int, default=4)
    parser.add_argument("--deep-branch-topk", type=int, default=2)
    parser.add_argument("--tail-branch-topk", type=int, default=1)
    parser.add_argument("--terminal-topk", type=int, default=2)
    parser.add_argument("--deep-terminal-topk", type=int, default=1)
    parser.add_argument("--teacher-budget-sec", type=float, default=120.0)
    parser.add_argument("--teacher-aig-timeout-sec", type=float, default=20.0)
    parser.add_argument("--teacher-final-timeout-sec", type=float, default=12.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    case_names = load_split(args.split, args.split_name)
    actions = default_actions()
    action_names = [action.name for action in actions]

    probe_case = case_names[0]
    env = AIGEnv(
        input_aig=args.case_root / probe_case / f"{probe_case}.aig",
        imap_bin=args.imap_bin,
        actions=actions,
        max_steps=args.max_steps,
        timeout_sec=args.timeout_sec,
    )
    obs_dim = int(env.reset().shape[0])
    action_dim = len(actions)

    dataset: list[TeacherRecord] = []
    case_stats: list[dict[str, float | str]] = []
    case_errors: list[dict[str, str]] = []
    tasks: list[dict[str, object]] = []
    manager = None
    event_queue = None
    cache_hits = 0
    cache_misses = 0
    for case_name in case_names:
        cache_key = _cache_key(
            case_name,
            args.max_steps,
            args.branch_topk,
            args.deep_branch_topk,
            args.tail_branch_topk,
            args.terminal_topk,
            args.deep_terminal_topk,
            args.timeout_sec,
            args.teacher_budget_sec,
            args.teacher_aig_timeout_sec,
            args.teacher_final_timeout_sec,
            action_names,
        )
        cache_file = args.cache_dir / f"{case_name}_{cache_key}.npz"
        if not args.rebuild_cache:
            try:
                cache_path, cache_mode = _find_cache_file(args.cache_dir, case_name, cache_key)
                cached = _load_cached_records(cache_path) if cache_path is not None else None
            except Exception as exc:
                case_errors.append({"case_name": case_name, "error": f"bad cache: {exc}"})
                cached = None
            if cached is not None:
                if cache_mode == "legacy" and cache_path is not None and cache_path != cache_file:
                    try:
                        shutil.copy2(cache_path, cache_file)
                    except Exception:
                        pass
                dataset.extend(cached)
                case_stats.append({"case_name": case_name, "records": float(len(cached)), "source": f"cache-{cache_mode}"})
                cache_hits += 1
                continue
        cache_misses += 1
        tasks.append(
            {
                "case_name": case_name,
                "cache_file": str(cache_file),
                "cache_key": cache_key,
                "rebuild_cache": args.rebuild_cache,
                "input_aig": str(args.case_root / case_name / f"{case_name}.aig"),
                "imap_bin": str(args.imap_bin),
                "max_steps": args.max_steps,
                "timeout_sec": args.timeout_sec,
                "branch_topk": args.branch_topk,
                "deep_branch_topk": args.deep_branch_topk,
                "tail_branch_topk": args.tail_branch_topk,
                "terminal_topk": args.terminal_topk,
                "deep_terminal_topk": args.deep_terminal_topk,
                "teacher_budget_sec": args.teacher_budget_sec,
                "teacher_aig_timeout_sec": min(args.timeout_sec, args.teacher_aig_timeout_sec),
                "teacher_final_timeout_sec": min(args.timeout_sec, args.teacher_final_timeout_sec),
                "estimated_states": estimate_teacher_states(
                    args.max_steps,
                    args.branch_topk,
                    args.deep_branch_topk,
                    args.tail_branch_topk,
                ),
            }
        )

    print(f"teacher cache: hits={cache_hits} miss={cache_misses} rebuild={args.rebuild_cache}")

    if tasks:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        event_queue = manager.Queue()
        for task in tasks:
            task["event_queue"] = event_queue
        monitor = threading.Thread(target=_progress_monitor, args=(event_queue, len(tasks)), daemon=True)
        monitor.start()
        pool = ctx.Pool(processes=max(1, args.num_workers))
        try:
            iterator = pool.imap_unordered(_collect_case_records, tasks, chunksize=1)
            for result in iterator:
                if result.get("source") == "error":
                    case_errors.append(
                        {
                            "case_name": str(result["case_name"]),
                            "error": str(result.get("error", "unknown error")),
                        }
                    )
                    continue

                case_name = str(result["case_name"])
                source = str(result["source"])
                cache_file = Path(str(result["cache_file"]))
                try:
                    records = _load_cached_records(cache_file)
                except Exception as exc:
                    case_errors.append({"case_name": case_name, "error": f"cache reload failed: {exc}"})
                    continue
                if not records:
                    case_errors.append({"case_name": case_name, "error": "cache reload returned no records"})
                    continue
                dataset.extend(records)
                case_stats.append({"case_name": case_name, "records": float(len(records)), "source": source})
        except KeyboardInterrupt:
            print("teacher collection interrupted, terminating workers...")
            pool.terminate()
            pool.join()
            raise
        except Exception:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
        finally:
            try:
                event_queue.put(("all_done",))
            except Exception:
                pass
            monitor.join()

    if not dataset:
        raise SystemExit("no teacher records collected")

    obs = torch.tensor(np.stack([item.obs for item in dataset], axis=0), dtype=torch.float32, device=device)
    actions_target = torch.tensor([item.best_action for item in dataset], dtype=torch.int64, device=device)
    q_target = torch.tensor(np.stack([item.q_values for item in dataset], axis=0), dtype=torch.float32, device=device)
    action_mask = torch.tensor(np.stack([item.action_mask for item in dataset], axis=0), dtype=torch.float32, device=device)
    value_target = torch.tensor([item.target_value for item in dataset], dtype=torch.float32, device=device)
    area_target = torch.tensor([item.target_area for item in dataset], dtype=torch.float32, device=device)
    depth_target = torch.tensor([item.target_depth for item in dataset], dtype=torch.float32, device=device)

    model = GreedyQNet(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        q_sum = 0.0
        value_sum = 0.0
        area_sum = 0.0
        depth_sum = 0.0
        num_batches = 0
        for batch in minibatch_indices(obs.shape[0], args.batch_size, rng):
            idx = torch.tensor(batch, dtype=torch.int64, device=device)
            batch_obs = obs.index_select(0, idx)
            batch_actions = actions_target.index_select(0, idx)
            batch_q_target = q_target.index_select(0, idx)
            batch_action_mask = action_mask.index_select(0, idx)
            batch_value = value_target.index_select(0, idx)
            batch_area = area_target.index_select(0, idx)
            batch_depth = depth_target.index_select(0, idx)

            q_values, value_pred, area_pred, depth_pred = model(batch_obs)
            masked_logits = q_values.masked_fill(batch_action_mask <= 0.0, -1e9)
            ce_loss = nn.functional.cross_entropy(masked_logits, batch_actions)
            q_loss = masked_q_loss(q_values, batch_q_target, batch_action_mask)
            value_loss = nn.functional.smooth_l1_loss(value_pred, batch_value)
            area_loss = nn.functional.smooth_l1_loss(area_pred, batch_area)
            depth_loss = nn.functional.smooth_l1_loss(depth_pred, batch_depth)
            loss = ce_loss + q_loss + 0.25 * value_loss + 0.05 * area_loss + 0.05 * depth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            num_batches += 1
            loss_sum += float(loss.item())
            ce_sum += float(ce_loss.item())
            q_sum += float(q_loss.item())
            value_sum += float(value_loss.item())
            area_sum += float(area_loss.item())
            depth_sum += float(depth_loss.item())

        summary = {
            "epoch": epoch,
            "records": len(dataset),
            "loss": loss_sum / max(1, num_batches),
            "ce_loss": ce_sum / max(1, num_batches),
            "q_loss": q_sum / max(1, num_batches),
            "value_loss": value_sum / max(1, num_batches),
            "area_loss": area_sum / max(1, num_batches),
            "depth_loss": depth_sum / max(1, num_batches),
        }
        history.append(summary)
        print(
            f"epoch {epoch}/{args.epochs}: records={len(dataset)} "
            f"loss={summary['loss']:.6f} ce={summary['ce_loss']:.6f} "
            f"q={summary['q_loss']:.6f} value={summary['value_loss']:.6f} "
            f"area={summary['area_loss']:.6f} depth={summary['depth_loss']:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": {
                "train_args": vars(args),
                "actions": action_names,
                "records": len(dataset),
                "cases": len(case_stats),
                "failed_cases": len(case_errors),
            },
        },
        args.output,
    )
    if args.history_json is not None:
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        args.history_json.write_text(
            json.dumps({"history": history, "cases": case_stats, "errors": case_errors}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
