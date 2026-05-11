#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_flow.actions import default_macro_actions
from rl_flow.budget import adapt_search_budget, case_bucket, case_weight
from rl_flow.imap_env import ImapEnv
from rl_flow.model import PolicyValueNet
from rl_flow.policy_search import beam_search, load_policy_checkpoint, search_candidates
from rl_flow.progress import progress_iter
from rl_flow.utils import load_split, read_ref_qor, resolve_device


_SAMPLER_MODEL: PolicyValueNet | None = None
_SAMPLER_ACTIONS = None
_SAMPLER_CASE_ROOT: Path | None = None
_SAMPLER_IMAP_BIN: Path | None = None
_SAMPLER_MAX_STEPS = 4
_SAMPLER_TIMEOUT_SEC = 60.0
_SAMPLER_TEMPERATURE = 1.0
_SAMPLER_BEAM_WIDTH = 4
_SAMPLER_BRANCH_TOPK = 4
_SAMPLER_GUMBEL_SCALE = 0.0
_SAMPLER_RANDOM_MIX_PROB = 0.0
_SAMPLER_EXPAND_WORKERS = 1
_EVAL_MODEL: PolicyValueNet | None = None
_EVAL_ACTIONS = None
_EVAL_CASE_ROOT: Path | None = None
_EVAL_IMAP_BIN: Path | None = None
_EVAL_MAX_STEPS = 4
_EVAL_TIMEOUT_SEC = 60.0
_EVAL_BEAM_WIDTH = 5
_EVAL_BRANCH_TOPK = 4
_EVAL_TEMPERATURE = 1.0
_EVAL_EXPAND_WORKERS = 1


def _group_training_episodes(
    episodes: list[dict],
    archive: dict[str, dict],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode["case_name"])].append(episode)
    for case_name, item in archive.items():
        if item.get("steps"):
            grouped[str(case_name)].append(item)
    return grouped


def _state_key(case_name: str, obs: np.ndarray) -> tuple[str, bytes]:
    return case_name, np.asarray(obs, dtype=np.float32).tobytes()


def _selected_bucket_counts(bucket_rows: list[str], selected: list[int]) -> dict[str, int]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for idx in selected:
        counts[bucket_rows[idx]] += 1
    return counts


def build_policy_dataset(
    episodes: list[dict],
    archive: dict[str, dict],
    topk_per_case: int,
    rng: random.Random,
    small_threshold: float,
    large_threshold: float,
    balance_mode: str,
    weight_min: float,
    weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None:
    grouped = _group_training_episodes(episodes, archive)

    obs_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    weight_rows: list[float] = []
    bucket_rows: list[str] = []
    for case_name, case_episodes in grouped.items():
        del case_name
        ranked = sorted(case_episodes, key=lambda item: float(item["final_return"]), reverse=True)
        for episode in ranked[:topk_per_case]:
            init_cost = float(episode.get("initial_cost", episode["cost"]))
            quality_scale = 1.0 + max(0.0, float(episode["final_return"])) * 4.0
            weight = case_weight(init_cost, min_weight=weight_min, max_weight=weight_max) * quality_scale
            bucket = case_bucket(init_cost, small_threshold=small_threshold, large_threshold=large_threshold)
            for step in episode["steps"]:
                obs_rows.append(np.array(step.obs, copy=True))
                action_rows.append(int(step.action))
                weight_rows.append(weight)
                bucket_rows.append(bucket)

    if not obs_rows:
        return None

    selected = rebalance_indices(bucket_rows, rng, balance_mode)
    obs = torch.tensor(np.stack([obs_rows[i] for i in selected], axis=0), dtype=torch.float32)
    actions = torch.tensor([action_rows[i] for i in selected], dtype=torch.int64)
    weights = torch.tensor([weight_rows[i] for i in selected], dtype=torch.float32)
    return obs, actions, weights, _selected_bucket_counts(bucket_rows, selected)


def build_value_dataset(
    episodes: list[dict],
    archive: dict[str, dict],
    rng: random.Random,
    small_threshold: float,
    large_threshold: float,
    balance_mode: str,
    weight_min: float,
    weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None:
    grouped = _group_training_episodes(episodes, archive)
    state_values: dict[tuple[str, bytes], dict[str, object]] = {}
    for case_name, case_episodes in grouped.items():
        for episode in case_episodes:
            init_cost = float(episode.get("initial_cost", episode["cost"]))
            final_return = float(episode["final_return"])
            bucket = case_bucket(init_cost, small_threshold=small_threshold, large_threshold=large_threshold)
            weight = case_weight(init_cost, min_weight=weight_min, max_weight=weight_max)
            for step in episode["steps"]:
                obs = np.array(step.obs, copy=True)
                key = _state_key(case_name, obs)
                existing = state_values.get(key)
                if existing is None or final_return > float(existing["target"]):
                    state_values[key] = {
                        "obs": obs,
                        "target": final_return,
                        "weight": weight,
                        "bucket": bucket,
                    }

    if not state_values:
        return None

    obs_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    weight_rows: list[float] = []
    bucket_rows: list[str] = []
    for item in state_values.values():
        obs_rows.append(np.array(item["obs"], copy=True))
        target_rows.append(float(item["target"]))
        weight_rows.append(float(item["weight"]))
        bucket_rows.append(str(item["bucket"]))

    selected = rebalance_indices(bucket_rows, rng, balance_mode)
    obs = torch.tensor(np.stack([obs_rows[i] for i in selected], axis=0), dtype=torch.float32)
    targets = torch.tensor([target_rows[i] for i in selected], dtype=torch.float32)
    weights = torch.tensor([weight_rows[i] for i in selected], dtype=torch.float32)
    return obs, targets, weights, _selected_bucket_counts(bucket_rows, selected)


def build_action_ranking_dataset(
    decisions: list[dict],
    min_return_gap: float,
    max_pairs_per_bucket: int,
    rng: random.Random,
    small_threshold: float,
    large_threshold: float,
    balance_mode: str,
    weight_min: float,
    weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None:
    obs_rows: list[np.ndarray] = []
    better_action_rows: list[int] = []
    worse_action_rows: list[int] = []
    pair_weight_rows: list[float] = []
    bucket_rows: list[str] = []
    for item in decisions:
        init_cost = float(item["initial_cost"])
        scored_actions = [(int(action), float(score)) for action, score in item["action_scores"]]
        if len(scored_actions) < 2:
            continue
        ranked = sorted(scored_actions, key=lambda value: value[1], reverse=True)
        pair_bucket = case_bucket(
            init_cost,
            small_threshold=small_threshold,
            large_threshold=large_threshold,
        )
        pair_base_weight = case_weight(init_cost, min_weight=weight_min, max_weight=weight_max)
        pair_count = 0
        for better_idx in range(len(ranked)):
            for worse_idx in range(better_idx + 1, len(ranked)):
                better_action, better_return = ranked[better_idx]
                worse_action, worse_return = ranked[worse_idx]
                gap = better_return - worse_return
                if gap < min_return_gap:
                    continue
                pair_weight = pair_base_weight * min(4.0, 1.0 + gap / max(min_return_gap, 1e-6))
                obs_rows.append(np.array(item["obs"], copy=True))
                better_action_rows.append(better_action)
                worse_action_rows.append(worse_action)
                pair_weight_rows.append(pair_weight)
                bucket_rows.append(pair_bucket)
                pair_count += 1
                if pair_count >= max_pairs_per_bucket:
                    break
            if pair_count >= max_pairs_per_bucket:
                break

    if not obs_rows:
        return None

    selected = rebalance_indices(bucket_rows, rng, balance_mode)
    obs = torch.tensor(np.stack([obs_rows[i] for i in selected], axis=0), dtype=torch.float32)
    better_actions = torch.tensor([better_action_rows[i] for i in selected], dtype=torch.int64)
    worse_actions = torch.tensor([worse_action_rows[i] for i in selected], dtype=torch.int64)
    weights = torch.tensor([pair_weight_rows[i] for i in selected], dtype=torch.float32)
    return obs, better_actions, worse_actions, weights, _selected_bucket_counts(bucket_rows, selected)


def rebalance_indices(bucket_rows: list[str], rng: random.Random, balance_mode: str) -> list[int]:
    if balance_mode == "none" or not bucket_rows:
        return list(range(len(bucket_rows)))

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, bucket in enumerate(bucket_rows):
        grouped[bucket].append(idx)
    nonempty = [indices for indices in grouped.values() if indices]
    if not nonempty:
        return list(range(len(bucket_rows)))

    if balance_mode == "mean":
        target = max(1, int(round(sum(len(indices) for indices in nonempty) / len(nonempty))))
    else:
        target = max(len(indices) for indices in nonempty)

    selected: list[int] = []
    for bucket in ("small", "medium", "large"):
        indices = grouped.get(bucket, [])
        if not indices:
            continue
        if len(indices) >= target:
            selected.extend(rng.sample(indices, target))
        else:
            chosen = list(indices)
            while len(chosen) < target:
                chosen.append(rng.choice(indices))
            selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def summarize_bucket_metrics(
    items: list[dict],
    small_threshold: float,
    large_threshold: float,
) -> dict[str, dict[str, float]]:
    buckets = {
        "small": {"count": 0.0, "return_sum": 0.0, "cost_sum": 0.0, "ref_gap_sum": 0.0, "fail_sum": 0.0},
        "medium": {"count": 0.0, "return_sum": 0.0, "cost_sum": 0.0, "ref_gap_sum": 0.0, "fail_sum": 0.0},
        "large": {"count": 0.0, "return_sum": 0.0, "cost_sum": 0.0, "ref_gap_sum": 0.0, "fail_sum": 0.0},
    }
    for item in items:
        init_cost = float(item.get("initial_cost", item.get("cost", 0.0)))
        bucket = case_bucket(init_cost, small_threshold=small_threshold, large_threshold=large_threshold)
        entry = buckets[bucket]
        entry["count"] += 1.0
        entry["return_sum"] += float(item.get("final_return", 0.0))
        entry["cost_sum"] += float(item.get("cost", 0.0))
        entry["ref_gap_sum"] += float(item.get("ref_cost_gap", 0.0))
        entry["fail_sum"] += 1.0 if item.get("done_reason") in {"action_timeout", "action_error"} else 0.0
    for entry in buckets.values():
        count = max(1.0, entry["count"])
        entry["avg_return"] = entry["return_sum"] / count
        entry["avg_cost"] = entry["cost_sum"] / count
        entry["avg_ref_gap"] = entry["ref_gap_sum"] / count
        entry["fail_pct"] = 100.0 * entry["fail_sum"] / count
    return buckets


def minibatch_indices(size: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    indices = list(range(size))
    rng.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, size, batch_size)]


def _init_sample_worker(
    checkpoint_path: str,
    case_root: str,
    imap_bin: str,
    max_steps: int,
    timeout_sec: float,
    temperature: float,
    beam_width: int,
    branch_topk: int,
    gumbel_scale: float,
    random_mix_prob: float,
    expand_workers: int,
) -> None:
    global _SAMPLER_MODEL
    global _SAMPLER_ACTIONS
    global _SAMPLER_CASE_ROOT
    global _SAMPLER_IMAP_BIN
    global _SAMPLER_MAX_STEPS
    global _SAMPLER_TIMEOUT_SEC
    global _SAMPLER_TEMPERATURE
    global _SAMPLER_BEAM_WIDTH
    global _SAMPLER_BRANCH_TOPK
    global _SAMPLER_GUMBEL_SCALE
    global _SAMPLER_RANDOM_MIX_PROB
    global _SAMPLER_EXPAND_WORKERS

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    _SAMPLER_ACTIONS = default_macro_actions()
    _SAMPLER_MODEL, _ = load_policy_checkpoint(Path(checkpoint_path), torch.device("cpu"))
    _SAMPLER_CASE_ROOT = Path(case_root)
    _SAMPLER_IMAP_BIN = Path(imap_bin)
    _SAMPLER_MAX_STEPS = max_steps
    _SAMPLER_TIMEOUT_SEC = timeout_sec
    _SAMPLER_TEMPERATURE = temperature
    _SAMPLER_BEAM_WIDTH = beam_width
    _SAMPLER_BRANCH_TOPK = branch_topk
    _SAMPLER_GUMBEL_SCALE = gumbel_scale
    _SAMPLER_RANDOM_MIX_PROB = random_mix_prob
    _SAMPLER_EXPAND_WORKERS = expand_workers


def _sample_case_worker(case_name: str, episodes_per_case: int, seed: int) -> dict[str, list[dict]]:
    if _SAMPLER_MODEL is None or _SAMPLER_CASE_ROOT is None or _SAMPLER_IMAP_BIN is None:
        raise RuntimeError("sample worker not initialized")

    rng = random.Random(seed)
    aig_path = _SAMPLER_CASE_ROOT / case_name / f"{case_name}.aig"
    env = ImapEnv(
        input_aig=aig_path,
        imap_bin=_SAMPLER_IMAP_BIN,
        actions=_SAMPLER_ACTIONS,
        max_steps=_SAMPLER_MAX_STEPS,
        timeout_sec=_SAMPLER_TIMEOUT_SEC,
    )
    env.reset()
    if env.initial_snapshot is None:
        raise RuntimeError("failed to initialize environment")
    init_cost = float(env.initial_snapshot.cost)
    max_steps, beam_width, branch_topk = adapt_search_budget(
        init_cost,
        _SAMPLER_MAX_STEPS,
        _SAMPLER_BEAM_WIDTH,
        _SAMPLER_BRANCH_TOPK,
    )
    env.max_steps = max_steps

    records: list[dict] = []
    decision_records: list[dict] = []
    candidates, decisions = search_candidates(
        env=env,
        model=_SAMPLER_MODEL,
        device=torch.device("cpu"),
        rng=rng,
        num_candidates=episodes_per_case,
        beam_width=beam_width,
        branch_topk=branch_topk,
        temperature=_SAMPLER_TEMPERATURE,
        gumbel_scale=_SAMPLER_GUMBEL_SCALE,
        random_mix_prob=_SAMPLER_RANDOM_MIX_PROB,
        expand_workers=_SAMPLER_EXPAND_WORKERS,
        reset_env=False,
    )
    for episode in candidates:
        records.append(
            {
                "case_name": case_name,
                "steps": episode.steps,
                "cost": episode.final_cost,
                "area": episode.final_area,
                "depth": episode.final_depth,
                "sequence": episode.final_sequence,
                "final_return": episode.final_return,
                "initial_cost": init_cost,
                "done_reason": episode.done_reason,
            }
        )
    for decision in decisions:
        decision_records.append(
            {
                "case_name": case_name,
                "initial_cost": init_cost,
                "obs": np.array(decision.obs, copy=True),
                "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
            }
        )
    return {"episodes": records, "decisions": decision_records}


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
    _EVAL_MODEL, _ = load_policy_checkpoint(Path(checkpoint_path), torch.device("cpu"))
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

    aig_path = _EVAL_CASE_ROOT / case_name / f"{case_name}.aig"
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
    result = beam_search(
        env=env,
        model=_EVAL_MODEL,
        device=torch.device("cpu"),
        beam_width=beam_width,
        branch_topk=branch_topk,
        temperature=_EVAL_TEMPERATURE,
        expand_workers=_EVAL_EXPAND_WORKERS,
        reset_env=False,
    )
    result["case_name"] = case_name
    case_dir = _EVAL_CASE_ROOT / case_name
    ref = read_ref_qor(case_dir)
    result["ref"] = ref
    result["initial_cost"] = init_cost
    if ref is not None:
        ref_cost = 0.6 * ref["level"] + 0.4 * ref["area"]
        result["ref_cost_gap"] = float(result["cost"]) - ref_cost
        result["ref_area_gap"] = int(result["area"]) - ref["area"]
        result["ref_depth_gap"] = int(result["depth"]) - ref["level"]
    return result


def collect_episodes_parallel(
    case_names: list[str],
    checkpoint_path: Path,
    case_root: Path,
    imap_bin: Path,
    max_steps: int,
    timeout_sec: float,
    episodes_per_case: int,
    temperature: float,
    num_workers: int,
    seed: int,
    beam_width: int,
    branch_topk: int,
    gumbel_scale: float,
    random_mix_prob: float,
    expand_workers: int,
    mp_start_method: str,
) -> tuple[list[dict], list[dict]]:
    ctx = mp.get_context(mp_start_method)
    collected: list[dict] = []
    decisions: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_sample_worker,
        initargs=(
            str(checkpoint_path),
            str(case_root),
            str(imap_bin),
            max_steps,
            timeout_sec,
            temperature,
            beam_width,
            branch_topk,
            gumbel_scale,
            random_mix_prob,
            expand_workers,
        ),
    ) as executor:
        futures = []
        for case_index, case_name in enumerate(case_names):
            case_seed = seed + case_index * 100003
            futures.append(
                executor.submit(
                    _sample_case_worker,
                    case_name,
                    episodes_per_case,
                    case_seed,
                )
            )
        for future in progress_iter(
            as_completed(futures),
            total=len(futures),
            desc="collect",
            unit="case",
        ):
            try:
                result = future.result()
                collected.extend(result["episodes"])
                decisions.extend(result["decisions"])
            except Exception as exc:
                print(f"collect worker failed: {exc}")
    return collected, decisions


def evaluate_split(
    case_names: list[str],
    case_root: Path,
    imap_bin: Path,
    actions,
    model: PolicyValueNet,
    device: torch.device,
    max_steps: int,
    timeout_sec: float,
    beam_width: int,
    branch_topk: int,
    temperature: float,
    expand_workers: int,
    num_workers: int,
    small_threshold: float,
    large_threshold: float,
    checkpoint_path: Path | None = None,
    mp_start_method: str = "fork",
) -> dict[str, object]:
    results = []
    total_cost = 0.0
    ref_gap_sum = 0.0
    ref_area_gap_sum = 0.0
    ref_depth_gap_sum = 0.0
    ref_count = 0
    exact_ref_matches = 0
    timeout_count = 0
    error_count = 0
    model.eval()

    if num_workers > 1 and len(case_names) > 1:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for parallel evaluation")
        ctx = mp.get_context(mp_start_method)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_eval_worker,
            initargs=(
                str(checkpoint_path),
                str(case_root),
                str(imap_bin),
                max_steps,
                timeout_sec,
                beam_width,
                branch_topk,
                temperature,
                expand_workers,
            ),
        ) as executor:
            futures = [executor.submit(_evaluate_case_worker, case_name) for case_name in case_names]
            for future in progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="eval",
                unit="case",
            ):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"eval worker failed: {exc}")
                    continue
                results.append(result)
                total_cost += float(result["cost"])
                ref = result.get("ref")
                if result.get("done_reason") == "action_timeout":
                    timeout_count += 1
                elif result.get("done_reason") == "action_error":
                    error_count += 1
                if ref is not None:
                    ref_count += 1
                    ref_gap_sum += float(result["ref_cost_gap"])
                    ref_area_gap_sum += float(result["ref_area_gap"])
                    ref_depth_gap_sum += float(result["ref_depth_gap"])
                    if int(result["area"]) == ref["area"] and int(result["depth"]) == ref["level"]:
                        exact_ref_matches += 1
    else:
        for case_name in progress_iter(case_names, desc="eval", unit="case"):
            aig_path = case_root / case_name / f"{case_name}.aig"
            env = ImapEnv(
                input_aig=aig_path,
                imap_bin=imap_bin,
                actions=actions,
                max_steps=max_steps,
                timeout_sec=timeout_sec,
            )
            env.reset()
            if env.initial_snapshot is None:
                raise RuntimeError("failed to initialize environment")
            init_cost = float(env.initial_snapshot.cost)
            local_max_steps, local_beam_width, local_branch_topk = adapt_search_budget(
                init_cost,
                max_steps,
                beam_width,
                branch_topk,
            )
            env.max_steps = local_max_steps
            result = beam_search(
                env=env,
                model=model,
                device=device,
                beam_width=local_beam_width,
                branch_topk=local_branch_topk,
                temperature=temperature,
                expand_workers=expand_workers,
                reset_env=False,
            )
            result["case_name"] = case_name
            result["initial_cost"] = init_cost
            case_dir = case_root / case_name
            ref = read_ref_qor(case_dir)
            result["ref"] = ref
            if result.get("done_reason") == "action_timeout":
                timeout_count += 1
            elif result.get("done_reason") == "action_error":
                error_count += 1
            if ref is not None:
                ref_cost = 0.6 * ref["level"] + 0.4 * ref["area"]
                result["ref_cost_gap"] = float(result["cost"]) - ref_cost
                result["ref_area_gap"] = int(result["area"]) - ref["area"]
                result["ref_depth_gap"] = int(result["depth"]) - ref["level"]
                ref_count += 1
                ref_gap_sum += float(result["ref_cost_gap"])
                ref_area_gap_sum += float(result["ref_area_gap"])
                ref_depth_gap_sum += float(result["ref_depth_gap"])
                if int(result["area"]) == ref["area"] and int(result["depth"]) == ref["level"]:
                    exact_ref_matches += 1
            results.append(result)
            total_cost += float(result["cost"])

    bucket_stats = summarize_bucket_metrics(
        results,
        small_threshold=small_threshold,
        large_threshold=large_threshold,
    )
    return {
        "case_count": len(results),
        "avg_cost": total_cost / max(1, len(results)),
        "avg_ref_gap": ref_gap_sum / max(1, ref_count),
        "avg_ref_area_gap": ref_area_gap_sum / max(1, ref_count),
        "avg_ref_depth_gap": ref_depth_gap_sum / max(1, ref_count),
        "ref_count": ref_count,
        "exact_ref_matches": exact_ref_matches,
        "timeout_pct": 100.0 * timeout_count / max(1, len(results)),
        "error_pct": 100.0 * error_count / max(1, len(results)),
        "fail_pct": 100.0 * (timeout_count + error_count) / max(1, len(results)),
        "bucket_stats": bucket_stats,
        "results": results,
    }


def save_checkpoint(
    path: Path,
    model: PolicyValueNet,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    archive: dict[str, dict],
    extra: dict[str, object],
) -> None:
    serializable_archive = {}
    for case_name, item in archive.items():
        serializable_archive[case_name] = {
            "cost": float(item["cost"]),
            "area": int(item["area"]),
            "depth": int(item["depth"]),
            "sequence": str(item["sequence"]),
            "final_return": float(item["final_return"]),
            "initial_cost": float(item.get("initial_cost", item["cost"])),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "archive": serializable_archive,
            "meta": extra,
        },
        path,
    )


def save_sampling_checkpoint(
    path: Path,
    model: PolicyValueNet,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a policy/value model directly on final QoR.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--eval-split-name", choices=("train", "eval", "test"), default="eval")
    parser.add_argument("--case-root", type=Path, default=Path("/home/pan/eda/23_question2/iMAP/eda23/benchmark_public"))
    parser.add_argument("--imap-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes-per-case", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ranking-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--imitation-coef", type=float, default=0.4)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--min-return-gap", type=float, default=0.01)
    parser.add_argument("--max-pairs-per-bucket", type=int, default=8)
    parser.add_argument("--elite-topk-per-case", type=int, default=2)
    parser.add_argument("--small-cost-threshold", type=float, default=200.0)
    parser.add_argument("--large-cost-threshold", type=float, default=1000.0)
    parser.add_argument("--case-weight-min", type=float, default=1.0)
    parser.add_argument("--case-weight-max", type=float, default=1.8)
    parser.add_argument("--bucket-balance-mode", choices=("none", "mean", "max"), default="mean")
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--epsilon-decay", type=float, default=0.96)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--train-beam-width", type=int, default=4)
    parser.add_argument("--train-branch-topk", type=int, default=4)
    parser.add_argument("--train-search-workers", type=int, default=1)
    parser.add_argument("--search-gumbel-scale", type=float, default=0.4)
    parser.add_argument("--search-random-mix-prob", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--beam-branch-topk", type=int, default=4)
    parser.add_argument("--eval-search-workers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mp-start-method", choices=("fork", "spawn", "forkserver", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_cases = load_split(args.split, args.split_name)
    eval_cases = load_split(args.split, args.eval_split_name) if args.eval_split_name else []
    actions = default_macro_actions()

    probe_env = ImapEnv(
        input_aig=args.case_root / train_cases[0] / f"{train_cases[0]}.aig",
        imap_bin=args.imap_bin,
        actions=actions,
        max_steps=args.max_steps,
        timeout_sec=args.timeout_sec,
    )
    obs_dim = int(probe_env.reset().shape[0])
    action_dim = len(actions)

    model = PolicyValueNet(obs_dim, action_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_workers = args.num_workers if args.num_workers > 0 else min(8, os.cpu_count() or 1)
    eval_workers = args.eval_workers if args.eval_workers > 0 else num_workers
    mp_start_method = args.mp_start_method
    if mp_start_method == "auto":
        mp_start_method = "spawn" if device.type == "cuda" else "fork"

    print(f"train device: {device}")
    print(f"collect workers: {num_workers}")
    print(f"eval workers: {eval_workers}")
    print(f"mp start method: {mp_start_method}")

    history = []
    archive: dict[str, dict] = {}
    best_eval_cost = float("inf")
    epsilon = args.epsilon

    for epoch in range(1, args.epochs + 1):
        model.eval()
        sampling_ckpt = args.output.with_suffix(f".epoch{epoch}.sample.pt")
        save_sampling_checkpoint(
            sampling_ckpt,
            model,
            obs_dim,
            action_dim,
            args.hidden_dim,
        )
        try:
            if num_workers > 1 and len(train_cases) > 1:
                collected, collected_decisions = collect_episodes_parallel(
                    case_names=train_cases,
                    checkpoint_path=sampling_ckpt,
                    case_root=args.case_root,
                    imap_bin=args.imap_bin,
                    max_steps=args.max_steps,
                    timeout_sec=args.timeout_sec,
                    episodes_per_case=args.episodes_per_case,
                    temperature=args.temperature,
                    num_workers=num_workers,
                    seed=args.seed + epoch * 1000003,
                    beam_width=args.train_beam_width,
                    branch_topk=args.train_branch_topk,
                    gumbel_scale=args.search_gumbel_scale,
                    random_mix_prob=args.search_random_mix_prob,
                    expand_workers=args.train_search_workers,
                    mp_start_method=mp_start_method,
                )
            else:
                collected = []
                collected_decisions = []
                for case_index, case_name in enumerate(
                    progress_iter(train_cases, desc=f"collect epoch {epoch}", unit="case")
                ):
                    aig_path = args.case_root / case_name / f"{case_name}.aig"
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
                        args.train_beam_width,
                        args.train_branch_topk,
                    )
                    env.max_steps = max_steps
                    local_rng = random.Random(args.seed + epoch * 1000003 + case_index * 100003)
                    candidates, decisions = search_candidates(
                        env=env,
                        model=model,
                        device=device,
                        rng=local_rng,
                        num_candidates=args.episodes_per_case,
                        beam_width=beam_width,
                        branch_topk=branch_topk,
                        temperature=args.temperature,
                        gumbel_scale=args.search_gumbel_scale,
                        random_mix_prob=args.search_random_mix_prob,
                        expand_workers=args.train_search_workers,
                        reset_env=False,
                    )
                    for episode in candidates:
                        collected.append(
                            {
                                "case_name": case_name,
                                "steps": episode.steps,
                                "cost": episode.final_cost,
                                "area": episode.final_area,
                                "depth": episode.final_depth,
                                "sequence": episode.final_sequence,
                                "final_return": episode.final_return,
                                "initial_cost": init_cost,
                                "done_reason": episode.done_reason,
                            }
                        )
                    for decision in decisions:
                        collected_decisions.append(
                            {
                                "case_name": case_name,
                                "initial_cost": init_cost,
                                "obs": np.array(decision.obs, copy=True),
                                "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                            }
                        )
        finally:
            if sampling_ckpt.exists():
                sampling_ckpt.unlink()

        for record in collected:
            best = archive.get(record["case_name"])
            if best is None or float(record["cost"]) < float(best["cost"]):
                archive[record["case_name"]] = record
        epsilon = max(args.min_epsilon, epsilon * args.epsilon_decay)

        if not collected:
            raise SystemExit("no training episodes collected")

        policy_dataset = build_policy_dataset(
            episodes=collected,
            archive=archive,
            topk_per_case=args.elite_topk_per_case,
            rng=rng,
            small_threshold=args.small_cost_threshold,
            large_threshold=args.large_cost_threshold,
            balance_mode=args.bucket_balance_mode,
            weight_min=args.case_weight_min,
            weight_max=args.case_weight_max,
        )
        value_dataset = build_value_dataset(
            episodes=collected,
            archive=archive,
            rng=rng,
            small_threshold=args.small_cost_threshold,
            large_threshold=args.large_cost_threshold,
            balance_mode=args.bucket_balance_mode,
            weight_min=args.case_weight_min,
            weight_max=args.case_weight_max,
        )
        ranking_dataset = build_action_ranking_dataset(
            decisions=collected_decisions,
            min_return_gap=args.min_return_gap,
            max_pairs_per_bucket=args.max_pairs_per_bucket,
            rng=rng,
            small_threshold=args.small_cost_threshold,
            large_threshold=args.large_cost_threshold,
            balance_mode=args.bucket_balance_mode,
            weight_min=args.case_weight_min,
            weight_max=args.case_weight_max,
        )

        model.train()
        update_losses = []
        ranking_loss_value = 0.0
        value_loss_value = 0.0
        imitation_loss_value = 0.0
        policy_bucket_counts = {"small": 0, "medium": 0, "large": 0}
        value_bucket_counts = {"small": 0, "medium": 0, "large": 0}
        ranking_bucket_counts = {"small": 0, "medium": 0, "large": 0}

        policy_obs = None
        policy_actions = None
        policy_weights = None
        if policy_dataset is not None:
            policy_obs, policy_actions, policy_weights, policy_bucket_counts = policy_dataset
            policy_obs = policy_obs.to(device)
            policy_actions = policy_actions.to(device)
            policy_weights = policy_weights.to(device)

        value_obs = None
        value_targets = None
        value_weights = None
        if value_dataset is not None:
            value_obs, value_targets, value_weights, value_bucket_counts = value_dataset
            value_obs = value_obs.to(device)
            value_targets = value_targets.to(device)
            value_weights = value_weights.to(device)

        ranking_obs = None
        better_actions = None
        worse_actions = None
        ranking_weights = None
        if ranking_dataset is not None:
            ranking_obs, better_actions, worse_actions, ranking_weights, ranking_bucket_counts = ranking_dataset
            ranking_obs = ranking_obs.to(device)
            better_actions = better_actions.to(device)
            worse_actions = worse_actions.to(device)
            ranking_weights = ranking_weights.to(device)

        for _ in progress_iter(range(args.update_epochs), desc=f"update epoch {epoch}", unit="pass", leave=False):
            if value_obs is not None and args.value_coef > 0:
                for batch in minibatch_indices(value_obs.shape[0], args.batch_size, rng):
                    batch_idx = torch.tensor(batch, dtype=torch.int64, device=device)
                    batch_obs = value_obs.index_select(0, batch_idx)
                    batch_targets = value_targets.index_select(0, batch_idx)
                    batch_weights = value_weights.index_select(0, batch_idx)
                    _logits, values = model(batch_obs)
                    value_loss = nn.functional.smooth_l1_loss(values, batch_targets, reduction="none")
                    value_loss = (value_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
                    loss = args.value_coef * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    value_loss_value = float(value_loss.item())
                    update_losses.append(float(loss.item()))

            if ranking_obs is not None and args.ranking_coef > 0:
                for batch in minibatch_indices(ranking_obs.shape[0], args.batch_size, rng):
                    batch_idx = torch.tensor(batch, dtype=torch.int64, device=device)
                    batch_obs = ranking_obs.index_select(0, batch_idx)
                    batch_better_actions = better_actions.index_select(0, batch_idx)
                    batch_worse_actions = worse_actions.index_select(0, batch_idx)
                    batch_weights = ranking_weights.index_select(0, batch_idx)
                    logits, _values = model(batch_obs)
                    better_logits = logits.gather(1, batch_better_actions.unsqueeze(1)).squeeze(1)
                    worse_logits = logits.gather(1, batch_worse_actions.unsqueeze(1)).squeeze(1)
                    delta = better_logits - worse_logits
                    ranking_loss = nn.functional.softplus(args.ranking_margin - delta)
                    ranking_loss = (ranking_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
                    loss = args.ranking_coef * ranking_loss

                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    ranking_loss_value = float(ranking_loss.item())
                    update_losses.append(float(loss.item()))

            if policy_obs is not None and args.imitation_coef > 0:
                for batch in minibatch_indices(policy_obs.shape[0], args.batch_size, rng):
                    batch_idx = torch.tensor(batch, dtype=torch.int64, device=device)
                    batch_obs = policy_obs.index_select(0, batch_idx)
                    batch_actions = policy_actions.index_select(0, batch_idx)
                    batch_weights = policy_weights.index_select(0, batch_idx)
                    logits, _values = model(batch_obs)
                    imitation_loss = nn.functional.cross_entropy(logits, batch_actions, reduction="none")
                    imitation_loss = (imitation_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
                    loss = args.imitation_coef * imitation_loss

                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    imitation_loss_value = float(imitation_loss.item())
                    update_losses.append(float(loss.item()))

        train_avg_cost = sum(float(item["cost"]) for item in collected) / len(collected)
        train_best_cost = min(float(item["cost"]) for item in collected)
        train_avg_return = sum(float(item["final_return"]) for item in collected) / len(collected)
        train_timeout_count = sum(1 for item in collected if item.get("done_reason") == "action_timeout")
        train_error_count = sum(1 for item in collected if item.get("done_reason") == "action_error")
        train_fail_count = train_timeout_count + train_error_count
        train_bucket_stats = summarize_bucket_metrics(
            collected,
            small_threshold=args.small_cost_threshold,
            large_threshold=args.large_cost_threshold,
        )
        summary = {
            "epoch": epoch,
            "train_avg_cost": train_avg_cost,
            "train_best_cost": train_best_cost,
            "train_avg_return": train_avg_return,
            "train_timeout_pct": 100.0 * train_timeout_count / max(1, len(collected)),
            "train_error_pct": 100.0 * train_error_count / max(1, len(collected)),
            "train_fail_pct": 100.0 * train_fail_count / max(1, len(collected)),
            "epsilon": epsilon,
            "loss": sum(update_losses) / max(1, len(update_losses)),
            "ranking_loss": ranking_loss_value,
            "value_loss": value_loss_value,
            "imitation_loss": imitation_loss_value,
            "ranking_pairs": 0 if ranking_obs is None else int(ranking_obs.shape[0]),
            "value_examples": 0 if value_obs is None else int(value_obs.shape[0]),
            "policy_examples": 0 if policy_obs is None else int(policy_obs.shape[0]),
            "policy_bucket_counts": policy_bucket_counts,
            "value_bucket_counts": value_bucket_counts,
            "ranking_bucket_counts": ranking_bucket_counts,
            "train_bucket_stats": train_bucket_stats,
        }

        if eval_cases and epoch % args.eval_every == 0:
            eval_ckpt = args.output.with_suffix(f".epoch{epoch}.eval.pt")
            save_sampling_checkpoint(
                eval_ckpt,
                model,
                obs_dim,
                action_dim,
                args.hidden_dim,
            )
            try:
                eval_summary = evaluate_split(
                    case_names=eval_cases,
                    case_root=args.case_root,
                    imap_bin=args.imap_bin,
                    actions=actions,
                    model=model,
                    device=device,
                    max_steps=args.max_steps,
                    timeout_sec=args.timeout_sec,
                    beam_width=args.beam_width,
                    branch_topk=args.beam_branch_topk,
                    temperature=args.temperature,
                    expand_workers=args.eval_search_workers,
                    num_workers=eval_workers,
                    small_threshold=args.small_cost_threshold,
                    large_threshold=args.large_cost_threshold,
                    checkpoint_path=eval_ckpt,
                    mp_start_method=mp_start_method,
                )
            finally:
                if eval_ckpt.exists():
                    eval_ckpt.unlink()
            summary["eval_avg_cost"] = float(eval_summary["avg_cost"])
            summary["eval_avg_ref_gap"] = float(eval_summary["avg_ref_gap"])
            summary["eval_avg_ref_area_gap"] = float(eval_summary["avg_ref_area_gap"])
            summary["eval_avg_ref_depth_gap"] = float(eval_summary["avg_ref_depth_gap"])
            summary["eval_exact_ref_matches"] = int(eval_summary["exact_ref_matches"])
            summary["eval_fail_pct"] = float(eval_summary["fail_pct"])
            summary["eval_timeout_pct"] = float(eval_summary["timeout_pct"])
            summary["eval_bucket_stats"] = eval_summary["bucket_stats"]
            if float(eval_summary["avg_cost"]) < best_eval_cost:
                best_eval_cost = float(eval_summary["avg_cost"])
                save_checkpoint(
                    args.output,
                    model,
                    obs_dim,
                    action_dim,
                    args.hidden_dim,
                    archive,
                    {"best_eval_cost": best_eval_cost, "epoch": epoch, "train_args": vars(args)},
                )

        if not eval_cases:
            save_checkpoint(
                args.output,
                model,
                obs_dim,
                action_dim,
                args.hidden_dim,
                archive,
                {"epoch": epoch, "train_args": vars(args)},
            )

        history.append(summary)
        status = (
            f"epoch {epoch}/{args.epochs}: "
            f"train_avg_return={train_avg_return:.4f} "
            f"train_avg_cost={train_avg_cost:.4f} "
            f"train_best_cost={train_best_cost:.4f} "
            f"train_fail_pct={summary['train_fail_pct']:.2f}% "
            f"train_timeout_pct={summary['train_timeout_pct']:.2f}% "
            f"loss={summary['loss']:.6f} "
            f"ranking_loss={summary['ranking_loss']:.6f} "
            f"value_loss={summary['value_loss']:.6f} "
            f"imitation_loss={summary['imitation_loss']:.6f}"
        )
        if "eval_avg_cost" in summary:
            status += (
                f" eval_avg_cost={summary['eval_avg_cost']:.4f}"
                f" eval_avg_ref_gap={summary['eval_avg_ref_gap']:.4f}"
                f" eval_fail_pct={summary['eval_fail_pct']:.2f}%"
                f" eval_timeout_pct={summary['eval_timeout_pct']:.2f}%"
                f" eval_exact_ref_matches={summary['eval_exact_ref_matches']}"
            )
        print(status)
        train_bucket_line = " | ".join(
            f"{bucket}:n={int(stats['count'])},ret={stats['avg_return']:.3f},fail={stats['fail_pct']:.1f}%"
            for bucket, stats in summary["train_bucket_stats"].items()
        )
        print(f"train buckets: {train_bucket_line}")
        print(
            "train datasets: "
            f"policy={summary['policy_examples']} {summary['policy_bucket_counts']} "
            f"value={summary['value_examples']} {summary['value_bucket_counts']} "
            f"ranking={summary['ranking_pairs']} {summary['ranking_bucket_counts']}"
        )
        if "eval_bucket_stats" in summary:
            eval_bucket_line = " | ".join(
                f"{bucket}:n={int(stats['count'])},gap={stats['avg_ref_gap']:.3f},fail={stats['fail_pct']:.1f}%"
                for bucket, stats in summary["eval_bucket_stats"].items()
            )
            print(f"eval buckets: {eval_bucket_line}")

    if args.history_json is not None:
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        args.history_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
