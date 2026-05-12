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
from rl_flow.budget import adapt_search_budget, case_bucket, case_bucket_index, case_weight
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
_SAMPLER_LABEL_TEMPERATURE = 0.7
_SAMPLER_LABEL_BEAM_WIDTH = 5
_SAMPLER_LABEL_BRANCH_TOPK = 4
_SAMPLER_LABEL_GUMBEL_SCALE = 0.0
_SAMPLER_LABEL_RANDOM_MIX_PROB = 0.0
_SAMPLER_LABEL_EXPAND_WORKERS = 1
_SAMPLER_SHARED_LABEL_SEARCH = True
_SAMPLER_SHARED_LABEL_USE_TEACHER_BUDGET = False
_SAMPLER_ENABLE_HARD_LABEL_SEARCH = True
_SAMPLER_HARD_LABEL_ROOT_GAP = 0.03
_SAMPLER_SMALL_THRESHOLD = 200.0
_SAMPLER_LARGE_THRESHOLD = 1000.0
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
_EVAL_SMALL_THRESHOLD = 200.0
_EVAL_LARGE_THRESHOLD = 1000.0


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


def estimate_case_cost_proxy(case_root: Path, case_name: str) -> float:
    ref = read_ref_qor(case_root / case_name)
    if ref is None:
        return 0.0
    return 0.6 * float(ref["level"]) + 0.4 * float(ref["area"])


def build_case_buckets(
    case_names: list[str],
    case_root: Path,
    small_threshold: float,
    large_threshold: float,
) -> dict[str, list[str]]:
    buckets = {"small": [], "medium": [], "large": []}
    for case_name in case_names:
        proxy_cost = estimate_case_cost_proxy(case_root, case_name)
        bucket = case_bucket(
            proxy_cost,
            small_threshold=small_threshold,
            large_threshold=large_threshold,
        )
        buckets[bucket].append(case_name)
    return buckets


def sample_cases_for_epoch(
    case_buckets: dict[str, list[str]],
    cases_per_epoch: int,
    rng: random.Random,
) -> tuple[list[str], dict[str, int]]:
    all_cases = [case_name for bucket in ("small", "medium", "large") for case_name in case_buckets.get(bucket, [])]
    if cases_per_epoch <= 0 or cases_per_epoch >= len(all_cases):
        return list(all_cases), {bucket: len(case_buckets.get(bucket, [])) for bucket in ("small", "medium", "large")}

    active_buckets = [bucket for bucket in ("large", "medium", "small") if case_buckets.get(bucket)]
    if not active_buckets:
        return [], {"small": 0, "medium": 0, "large": 0}

    target_counts = {bucket: 0 for bucket in ("small", "medium", "large")}
    base = max(1, cases_per_epoch // len(active_buckets))
    for bucket in active_buckets:
        target_counts[bucket] = min(len(case_buckets[bucket]), base)

    assigned = sum(target_counts.values())
    remaining = max(0, cases_per_epoch - assigned)
    while remaining > 0:
        progressed = False
        for bucket in ("large", "medium", "small"):
            capacity = len(case_buckets.get(bucket, [])) - target_counts[bucket]
            if capacity <= 0:
                continue
            target_counts[bucket] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    selected_cases: list[str] = []
    selected_counts = {"small": 0, "medium": 0, "large": 0}
    for bucket in ("small", "medium", "large"):
        cases = list(case_buckets.get(bucket, []))
        if not cases or target_counts[bucket] <= 0:
            continue
        chosen = rng.sample(cases, target_counts[bucket])
        selected_cases.extend(chosen)
        selected_counts[bucket] = len(chosen)
    rng.shuffle(selected_cases)
    return selected_cases, selected_counts


def decision_top_gap(action_scores: list[tuple[int, float]]) -> float:
    if len(action_scores) < 2:
        return 0.0
    ordered = sorted((float(score) for _action, score in action_scores), reverse=True)
    return float(ordered[0] - ordered[1])


def should_run_hard_label_search(
    init_cost: float,
    episodes: list[dict],
    decisions: list[dict] | list[object],
    small_threshold: float,
    large_threshold: float,
    hard_label_root_gap: float,
) -> bool:
    if case_bucket(init_cost, small_threshold=small_threshold, large_threshold=large_threshold) == "large":
        return True
    if any(item.get("done_reason") in {"action_timeout", "action_error"} for item in episodes):
        return True
    if decisions:
        first = decisions[0]
        scores = first["action_scores"] if isinstance(first, dict) else first.action_scores
        if decision_top_gap(scores) < hard_label_root_gap:
            return True
    return False


def build_tree_datasets(
    decisions: list[dict],
    rng: random.Random,
    small_threshold: float,
    large_threshold: float,
    balance_mode: str,
    weight_min: float,
    weight_max: float,
    policy_target_temperature: float,
    policy_min_gap: float,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None,
]:
    policy_states: dict[tuple[str, bytes], dict[str, object]] = {}
    value_states: dict[tuple[str, bytes], dict[str, object]] = {}
    source_priority = {"explore": 0, "teacher": 1}
    source_weight = {"explore": 1.0, "teacher": 1.35}

    for item in decisions:
        case_name = str(item["case_name"])
        init_cost = float(item["initial_cost"])
        bucket = case_bucket(init_cost, small_threshold=small_threshold, large_threshold=large_threshold)
        base_weight = case_weight(init_cost, min_weight=weight_min, max_weight=weight_max)
        source = str(item.get("source", "explore"))
        source_rank = source_priority.get(source, 0)
        source_scale = source_weight.get(source, 1.0)
        obs = np.asarray(item["obs"], dtype=np.float32)
        state_key = _state_key(case_name, obs)
        state_return = float(item["state_return"])

        value_entry = value_states.get(state_key)
        if (
            value_entry is None
            or state_return > float(value_entry["target"])
            or (
                state_return == float(value_entry["target"])
                and source_rank > int(value_entry.get("source_rank", 0))
            )
        ):
            value_states[state_key] = {
                "obs": np.array(obs, copy=True),
                "target": state_return,
                "weight": base_weight * source_scale * (1.0 + max(0.0, state_return) * 4.0),
                "bucket": bucket,
                "bucket_id": case_bucket_index(
                    init_cost,
                    small_threshold=small_threshold,
                    large_threshold=large_threshold,
                ),
                "source_rank": source_rank,
            }

        scored_actions = [(int(action), float(score)) for action, score in item["action_scores"]]
        if len(scored_actions) < 2:
            continue

        policy_entry = policy_states.get(state_key)
        if policy_entry is None:
            policy_entry = {
                "obs": np.array(obs, copy=True),
                "action_map": {},
                "target": state_return,
                "weight": base_weight * source_scale,
                "bucket": bucket,
                "bucket_id": case_bucket_index(
                    init_cost,
                    small_threshold=small_threshold,
                    large_threshold=large_threshold,
                ),
                "source_rank": source_rank,
            }
            policy_states[state_key] = policy_entry
        if source_rank > int(policy_entry.get("source_rank", 0)):
            policy_entry["obs"] = np.array(obs, copy=True)
            policy_entry["action_map"] = {}
            policy_entry["target"] = state_return
            policy_entry["weight"] = base_weight * source_scale
            policy_entry["bucket"] = bucket
            policy_entry["bucket_id"] = case_bucket_index(
                init_cost,
                small_threshold=small_threshold,
                large_threshold=large_threshold,
            )
            policy_entry["source_rank"] = source_rank
        if source_rank == int(policy_entry.get("source_rank", 0)):
            policy_entry["target"] = max(float(policy_entry["target"]), state_return)
            policy_entry["weight"] = max(float(policy_entry["weight"]), base_weight * source_scale)
            for action, score in scored_actions:
                current = policy_entry["action_map"].get(action)
                if current is None or score > float(current):
                    policy_entry["action_map"][action] = float(score)

    def build_policy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None:
        obs_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        weight_rows: list[float] = []
        bucket_rows: list[str] = []
        bucket_id_rows: list[int] = []
        for item in policy_states.values():
            action_map = item["action_map"]
            if len(action_map) < 2:
                continue
            actions = list(action_map.keys())
            scores = np.array([float(action_map[action]) for action in actions], dtype=np.float32)
            order = np.argsort(scores)[::-1]
            sorted_scores = scores[order]
            top_gap = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            if top_gap < policy_min_gap:
                continue
            logits = scores / max(1e-3, policy_target_temperature)
            logits -= float(np.max(logits))
            probs = np.exp(logits)
            probs_sum = float(probs.sum())
            if probs_sum <= 0:
                probs = np.full_like(probs, 1.0 / max(1, len(probs)))
            else:
                probs /= probs_sum
            target = np.zeros(len(default_macro_actions()), dtype=np.float32)
            for action, prob in zip(actions, probs.tolist()):
                target[action] = float(prob)
            weight = float(item["weight"]) * min(4.0, 1.0 + max(0.0, top_gap) / max(policy_target_temperature, 1e-3))
            obs_rows.append(np.array(item["obs"], copy=True))
            target_rows.append(target)
            weight_rows.append(weight)
            bucket_rows.append(str(item["bucket"]))
            bucket_id_rows.append(int(item["bucket_id"]))
        if not obs_rows:
            return None
        selected = rebalance_indices(bucket_rows, rng, balance_mode)
        obs = torch.tensor(np.stack([obs_rows[i] for i in selected], axis=0), dtype=torch.float32)
        target_tensor = torch.tensor(np.stack([target_rows[i] for i in selected], axis=0), dtype=torch.float32)
        weights = torch.tensor([weight_rows[i] for i in selected], dtype=torch.float32)
        bucket_ids = torch.tensor([bucket_id_rows[i] for i in selected], dtype=torch.int64)
        return obs, target_tensor, weights, bucket_ids, _selected_bucket_counts(bucket_rows, selected)

    def build_value() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | None:
        obs_rows: list[np.ndarray] = []
        target_rows: list[float] = []
        weight_rows: list[float] = []
        bucket_rows: list[str] = []
        for item in value_states.values():
            obs_rows.append(np.array(item["obs"], copy=True))
            target_rows.append(float(item["target"]))
            weight_rows.append(float(item["weight"]))
            bucket_rows.append(str(item["bucket"]))
        if not obs_rows:
            return None
        selected = rebalance_indices(bucket_rows, rng, balance_mode)
        obs = torch.tensor(np.stack([obs_rows[i] for i in selected], axis=0), dtype=torch.float32)
        targets = torch.tensor([target_rows[i] for i in selected], dtype=torch.float32)
        weights = torch.tensor([weight_rows[i] for i in selected], dtype=torch.float32)
        return obs, targets, weights, _selected_bucket_counts(bucket_rows, selected)

    return build_policy(), build_value()


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
    label_temperature: float,
    label_beam_width: int,
    label_branch_topk: int,
    label_gumbel_scale: float,
    label_random_mix_prob: float,
    label_expand_workers: int,
    shared_label_search: bool,
    shared_label_use_teacher_budget: bool,
    enable_hard_label_search: bool,
    hard_label_root_gap: float,
    small_threshold: float,
    large_threshold: float,
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
    global _SAMPLER_LABEL_TEMPERATURE
    global _SAMPLER_LABEL_BEAM_WIDTH
    global _SAMPLER_LABEL_BRANCH_TOPK
    global _SAMPLER_LABEL_GUMBEL_SCALE
    global _SAMPLER_LABEL_RANDOM_MIX_PROB
    global _SAMPLER_LABEL_EXPAND_WORKERS
    global _SAMPLER_SHARED_LABEL_SEARCH
    global _SAMPLER_SHARED_LABEL_USE_TEACHER_BUDGET
    global _SAMPLER_ENABLE_HARD_LABEL_SEARCH
    global _SAMPLER_HARD_LABEL_ROOT_GAP
    global _SAMPLER_SMALL_THRESHOLD
    global _SAMPLER_LARGE_THRESHOLD

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
    _SAMPLER_LABEL_TEMPERATURE = label_temperature
    _SAMPLER_LABEL_BEAM_WIDTH = label_beam_width
    _SAMPLER_LABEL_BRANCH_TOPK = label_branch_topk
    _SAMPLER_LABEL_GUMBEL_SCALE = label_gumbel_scale
    _SAMPLER_LABEL_RANDOM_MIX_PROB = label_random_mix_prob
    _SAMPLER_LABEL_EXPAND_WORKERS = label_expand_workers
    _SAMPLER_SHARED_LABEL_SEARCH = shared_label_search
    _SAMPLER_SHARED_LABEL_USE_TEACHER_BUDGET = shared_label_use_teacher_budget
    _SAMPLER_ENABLE_HARD_LABEL_SEARCH = enable_hard_label_search
    _SAMPLER_HARD_LABEL_ROOT_GAP = hard_label_root_gap
    _SAMPLER_SMALL_THRESHOLD = small_threshold
    _SAMPLER_LARGE_THRESHOLD = large_threshold


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
    if _SAMPLER_SHARED_LABEL_SEARCH and _SAMPLER_SHARED_LABEL_USE_TEACHER_BUDGET:
        beam_width = max(beam_width, _SAMPLER_LABEL_BEAM_WIDTH)
        branch_topk = max(branch_topk, _SAMPLER_LABEL_BRANCH_TOPK)
        temperature = _SAMPLER_LABEL_TEMPERATURE
        gumbel_scale = max(_SAMPLER_GUMBEL_SCALE, _SAMPLER_LABEL_GUMBEL_SCALE)
        random_mix_prob = max(_SAMPLER_RANDOM_MIX_PROB, _SAMPLER_LABEL_RANDOM_MIX_PROB)
        expand_workers = max(_SAMPLER_EXPAND_WORKERS, _SAMPLER_LABEL_EXPAND_WORKERS)
    else:
        temperature = _SAMPLER_TEMPERATURE
        gumbel_scale = _SAMPLER_GUMBEL_SCALE
        random_mix_prob = _SAMPLER_RANDOM_MIX_PROB
        expand_workers = _SAMPLER_EXPAND_WORKERS
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
        temperature=temperature,
        gumbel_scale=gumbel_scale,
        random_mix_prob=random_mix_prob,
        expand_workers=expand_workers,
        reset_env=False,
        small_threshold=_SAMPLER_SMALL_THRESHOLD,
        large_threshold=_SAMPLER_LARGE_THRESHOLD,
    )
    for episode in candidates:
        records.append(
            {
                "case_name": case_name,
                "source": "explore",
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
    hard_label_used = False
    if _SAMPLER_SHARED_LABEL_SEARCH:
        if _SAMPLER_ENABLE_HARD_LABEL_SEARCH and should_run_hard_label_search(
            init_cost=init_cost,
            episodes=records,
            decisions=decisions,
            small_threshold=_SAMPLER_SMALL_THRESHOLD,
            large_threshold=_SAMPLER_LARGE_THRESHOLD,
            hard_label_root_gap=_SAMPLER_HARD_LABEL_ROOT_GAP,
        ):
            hard_label_used = True
        else:
            for decision in decisions:
                decision_records.append(
                    {
                        "case_name": case_name,
                        "source": "teacher",
                        "initial_cost": init_cost,
                        "obs": np.array(decision.obs, copy=True),
                        "state_return": float(decision.state_return),
                        "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                    }
                )
    else:
        for decision in decisions:
            decision_records.append(
                {
                    "case_name": case_name,
                    "source": "explore",
                    "initial_cost": init_cost,
                    "obs": np.array(decision.obs, copy=True),
                    "state_return": float(decision.state_return),
                    "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                }
            )
        hard_label_used = True

    if hard_label_used:
        env.reset()
        if env.initial_snapshot is None:
            raise RuntimeError("failed to initialize environment")
        label_init_cost = float(env.initial_snapshot.cost)
        label_max_steps, label_beam_width, label_branch_topk = adapt_search_budget(
            label_init_cost,
            _SAMPLER_MAX_STEPS,
            _SAMPLER_LABEL_BEAM_WIDTH,
            _SAMPLER_LABEL_BRANCH_TOPK,
        )
        env.max_steps = label_max_steps
        shared_candidates, shared_decisions = search_candidates(
            env=env,
            model=_SAMPLER_MODEL,
            device=torch.device("cpu"),
            rng=random.Random(seed + 9173),
            num_candidates=1,
            beam_width=label_beam_width,
            branch_topk=label_branch_topk,
            temperature=_SAMPLER_LABEL_TEMPERATURE,
            gumbel_scale=_SAMPLER_LABEL_GUMBEL_SCALE,
            random_mix_prob=_SAMPLER_LABEL_RANDOM_MIX_PROB,
            expand_workers=_SAMPLER_LABEL_EXPAND_WORKERS,
            reset_env=False,
            small_threshold=_SAMPLER_SMALL_THRESHOLD,
            large_threshold=_SAMPLER_LARGE_THRESHOLD,
        )
        if shared_candidates:
            best_label = min(shared_candidates, key=lambda item: float(item.final_cost))
            records.append(
                {
                    "case_name": case_name,
                    "source": "teacher",
                    "steps": best_label.steps,
                    "cost": best_label.final_cost,
                    "area": best_label.final_area,
                    "depth": best_label.final_depth,
                    "sequence": best_label.final_sequence,
                    "final_return": best_label.final_return,
                    "initial_cost": label_init_cost,
                    "done_reason": best_label.done_reason,
                }
            )
        for decision in shared_decisions:
            decision_records.append(
                {
                    "case_name": case_name,
                    "source": "teacher",
                    "initial_cost": label_init_cost,
                    "obs": np.array(decision.obs, copy=True),
                    "state_return": float(decision.state_return),
                    "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                }
            )
    return {"episodes": records, "decisions": decision_records, "hard_label_used": hard_label_used}


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
    small_threshold: float,
    large_threshold: float,
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
    global _EVAL_SMALL_THRESHOLD
    global _EVAL_LARGE_THRESHOLD

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
    _EVAL_SMALL_THRESHOLD = small_threshold
    _EVAL_LARGE_THRESHOLD = large_threshold


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
        small_threshold=_EVAL_SMALL_THRESHOLD,
        large_threshold=_EVAL_LARGE_THRESHOLD,
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
    label_temperature: float,
    label_beam_width: int,
    label_branch_topk: int,
    label_gumbel_scale: float,
    label_random_mix_prob: float,
    label_expand_workers: int,
    shared_label_search: bool,
    shared_label_use_teacher_budget: bool,
    enable_hard_label_search: bool,
    hard_label_root_gap: float,
    small_threshold: float,
    large_threshold: float,
    mp_start_method: str,
) -> tuple[list[dict], list[dict], int]:
    ctx = mp.get_context(mp_start_method)
    collected: list[dict] = []
    decisions: list[dict] = []
    hard_label_count = 0
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
            label_temperature,
            label_beam_width,
            label_branch_topk,
            label_gumbel_scale,
            label_random_mix_prob,
            label_expand_workers,
            shared_label_search,
            shared_label_use_teacher_budget,
            enable_hard_label_search,
            hard_label_root_gap,
            small_threshold,
            large_threshold,
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
                hard_label_count += int(result.get("hard_label_used", False))
            except Exception as exc:
                print(f"collect worker failed: {exc}")
    return collected, decisions, hard_label_count


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
                small_threshold,
                large_threshold,
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
                small_threshold=small_threshold,
                large_threshold=large_threshold,
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
    num_buckets: int,
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
            "num_buckets": num_buckets,
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
    num_buckets: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "num_buckets": num_buckets,
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
    parser.add_argument("--cases-per-epoch", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--policy-coef", type=float, default=None)
    parser.add_argument("--ranking-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--imitation-coef", type=float, default=0.0)
    parser.add_argument("--policy-target-temperature", type=float, default=0.05)
    parser.add_argument("--policy-min-gap", type=float, default=0.01)
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
    parser.add_argument("--label-temperature", type=float, default=0.7)
    parser.add_argument("--label-beam-width", type=int, default=6)
    parser.add_argument("--label-branch-topk", type=int, default=5)
    parser.add_argument("--label-search-workers", type=int, default=1)
    parser.add_argument("--label-gumbel-scale", type=float, default=0.0)
    parser.add_argument("--label-random-mix-prob", type=float, default=0.0)
    parser.add_argument("--separate-label-search", action="store_true")
    parser.add_argument("--shared-label-use-teacher-budget", action="store_true")
    parser.add_argument("--disable-hard-label-search", action="store_true")
    parser.add_argument("--hard-label-root-gap", type=float, default=0.03)
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
    train_case_buckets = build_case_buckets(
        train_cases,
        args.case_root,
        args.small_cost_threshold,
        args.large_cost_threshold,
    )
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

    num_buckets = 3
    model = PolicyValueNet(obs_dim, action_dim, args.hidden_dim, num_buckets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_workers = args.num_workers if args.num_workers > 0 else min(8, os.cpu_count() or 1)
    eval_workers = args.eval_workers if args.eval_workers > 0 else num_workers
    mp_start_method = args.mp_start_method
    if mp_start_method == "auto":
        mp_start_method = "spawn" if device.type == "cuda" else "fork"
    shared_label_search = not args.separate_label_search
    shared_label_use_teacher_budget = args.shared_label_use_teacher_budget
    enable_hard_label_search = not args.disable_hard_label_search

    print(f"train device: {device}")
    print(f"collect workers: {num_workers}")
    print(f"eval workers: {eval_workers}")
    print(f"mp start method: {mp_start_method}")

    history = []
    archive: dict[str, dict] = {}
    best_eval_cost = float("inf")
    epsilon = args.epsilon

    for epoch in range(1, args.epochs + 1):
        epoch_rng = random.Random(args.seed + epoch * 2000003)
        epoch_train_cases, sampled_case_counts = sample_cases_for_epoch(
            train_case_buckets,
            args.cases_per_epoch,
            epoch_rng,
        )
        model.eval()
        sampling_ckpt = args.output.with_suffix(f".epoch{epoch}.sample.pt")
        save_sampling_checkpoint(
            sampling_ckpt,
            model,
            obs_dim,
            action_dim,
            args.hidden_dim,
            model.num_buckets,
        )
        try:
            if num_workers > 1 and len(epoch_train_cases) > 1:
                collected, collected_decisions, hard_label_case_count = collect_episodes_parallel(
                    case_names=epoch_train_cases,
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
                    label_temperature=args.label_temperature,
                    label_beam_width=args.label_beam_width,
                    label_branch_topk=args.label_branch_topk,
                    label_gumbel_scale=args.label_gumbel_scale,
                    label_random_mix_prob=args.label_random_mix_prob,
                    label_expand_workers=args.label_search_workers,
                    shared_label_search=shared_label_search,
                    shared_label_use_teacher_budget=shared_label_use_teacher_budget,
                    enable_hard_label_search=enable_hard_label_search,
                    hard_label_root_gap=args.hard_label_root_gap,
                    small_threshold=args.small_cost_threshold,
                    large_threshold=args.large_cost_threshold,
                    mp_start_method=mp_start_method,
                )
            else:
                collected = []
                collected_decisions = []
                hard_label_case_count = 0
                for case_index, case_name in enumerate(
                    progress_iter(epoch_train_cases, desc=f"collect epoch {epoch}", unit="case")
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
                    if shared_label_search and shared_label_use_teacher_budget:
                        beam_width = max(beam_width, args.label_beam_width)
                        branch_topk = max(branch_topk, args.label_branch_topk)
                        temperature = args.label_temperature
                        gumbel_scale = max(args.search_gumbel_scale, args.label_gumbel_scale)
                        random_mix_prob = max(args.search_random_mix_prob, args.label_random_mix_prob)
                        expand_workers = max(args.train_search_workers, args.label_search_workers)
                    else:
                        temperature = args.temperature
                        gumbel_scale = args.search_gumbel_scale
                        random_mix_prob = args.search_random_mix_prob
                        expand_workers = args.train_search_workers
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
                        temperature=temperature,
                        gumbel_scale=gumbel_scale,
                        random_mix_prob=random_mix_prob,
                        expand_workers=expand_workers,
                        reset_env=False,
                        small_threshold=args.small_cost_threshold,
                        large_threshold=args.large_cost_threshold,
                    )
                    for episode in candidates:
                        collected.append(
                            {
                                "case_name": case_name,
                                "source": "explore",
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
                    if shared_label_search:
                        if enable_hard_label_search and should_run_hard_label_search(
                            init_cost=init_cost,
                            episodes=collected[-len(candidates):],
                            decisions=decisions,
                            small_threshold=args.small_cost_threshold,
                            large_threshold=args.large_cost_threshold,
                            hard_label_root_gap=args.hard_label_root_gap,
                        ):
                            hard_label_case_count += 1
                            env.reset()
                            if env.initial_snapshot is None:
                                raise RuntimeError("failed to initialize environment")
                            label_init_cost = float(env.initial_snapshot.cost)
                            label_max_steps, label_beam_width, label_branch_topk = adapt_search_budget(
                                label_init_cost,
                                args.max_steps,
                                args.label_beam_width,
                                args.label_branch_topk,
                            )
                            env.max_steps = label_max_steps
                            label_rng = random.Random(args.seed + epoch * 1000003 + case_index * 100003 + 9173)
                            label_candidates, label_decisions = search_candidates(
                                env=env,
                                model=model,
                                device=device,
                                rng=label_rng,
                                num_candidates=1,
                                beam_width=label_beam_width,
                                branch_topk=label_branch_topk,
                                temperature=args.label_temperature,
                                gumbel_scale=args.label_gumbel_scale,
                                random_mix_prob=args.label_random_mix_prob,
                                expand_workers=args.label_search_workers,
                                reset_env=False,
                                small_threshold=args.small_cost_threshold,
                                large_threshold=args.large_cost_threshold,
                            )
                            if label_candidates:
                                best_label = min(label_candidates, key=lambda item: float(item.final_cost))
                                collected.append(
                                    {
                                        "case_name": case_name,
                                        "source": "teacher",
                                        "steps": best_label.steps,
                                        "cost": best_label.final_cost,
                                        "area": best_label.final_area,
                                        "depth": best_label.final_depth,
                                        "sequence": best_label.final_sequence,
                                        "final_return": best_label.final_return,
                                        "initial_cost": label_init_cost,
                                        "done_reason": best_label.done_reason,
                                    }
                                )
                            for decision in label_decisions:
                                collected_decisions.append(
                                    {
                                        "case_name": case_name,
                                        "source": "teacher",
                                        "initial_cost": label_init_cost,
                                        "obs": np.array(decision.obs, copy=True),
                                        "state_return": float(decision.state_return),
                                        "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                                    }
                                )
                        else:
                            for decision in decisions:
                                collected_decisions.append(
                                    {
                                        "case_name": case_name,
                                        "source": "teacher",
                                        "initial_cost": init_cost,
                                        "obs": np.array(decision.obs, copy=True),
                                        "state_return": float(decision.state_return),
                                        "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                                    }
                                )
                    else:
                        hard_label_case_count += 1
                        for decision in decisions:
                            collected_decisions.append(
                                {
                                    "case_name": case_name,
                                    "source": "explore",
                                    "initial_cost": init_cost,
                                    "obs": np.array(decision.obs, copy=True),
                                    "state_return": float(decision.state_return),
                                    "action_scores": [(int(action), float(score)) for action, score in decision.action_scores],
                                }
                            )
                        env.reset()
                        if env.initial_snapshot is None:
                            raise RuntimeError("failed to initialize environment")
                        label_init_cost = float(env.initial_snapshot.cost)
                        label_max_steps, label_beam_width, label_branch_topk = adapt_search_budget(
                            label_init_cost,
                            args.max_steps,
                            args.label_beam_width,
                            args.label_branch_topk,
                        )
                        env.max_steps = label_max_steps
                        label_rng = random.Random(args.seed + epoch * 1000003 + case_index * 100003 + 9173)
                        label_candidates, label_decisions = search_candidates(
                            env=env,
                            model=model,
                            device=device,
                            rng=label_rng,
                            num_candidates=1,
                            beam_width=label_beam_width,
                            branch_topk=label_branch_topk,
                            temperature=args.label_temperature,
                            gumbel_scale=args.label_gumbel_scale,
                            random_mix_prob=args.label_random_mix_prob,
                            expand_workers=args.label_search_workers,
                            reset_env=False,
                            small_threshold=args.small_cost_threshold,
                            large_threshold=args.large_cost_threshold,
                        )
                        if label_candidates:
                            best_label = min(label_candidates, key=lambda item: float(item.final_cost))
                            collected.append(
                                {
                                    "case_name": case_name,
                                    "source": "teacher",
                                    "steps": best_label.steps,
                                    "cost": best_label.final_cost,
                                    "area": best_label.final_area,
                                    "depth": best_label.final_depth,
                                    "sequence": best_label.final_sequence,
                                    "final_return": best_label.final_return,
                                    "initial_cost": label_init_cost,
                                    "done_reason": best_label.done_reason,
                                }
                            )
                        for decision in label_decisions:
                            collected_decisions.append(
                                {
                                    "case_name": case_name,
                                    "source": "teacher",
                                    "initial_cost": label_init_cost,
                                    "obs": np.array(decision.obs, copy=True),
                                    "state_return": float(decision.state_return),
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

        policy_dataset, value_dataset = build_tree_datasets(
            decisions=collected_decisions,
            rng=rng,
            small_threshold=args.small_cost_threshold,
            large_threshold=args.large_cost_threshold,
            balance_mode=args.bucket_balance_mode,
            weight_min=args.case_weight_min,
            weight_max=args.case_weight_max,
            policy_target_temperature=args.policy_target_temperature,
            policy_min_gap=args.policy_min_gap,
        )

        model.train()
        update_losses = []
        policy_loss_value = 0.0
        value_loss_value = 0.0
        imitation_loss_value = 0.0
        policy_bucket_counts = {"small": 0, "medium": 0, "large": 0}
        value_bucket_counts = {"small": 0, "medium": 0, "large": 0}
        policy_coef = args.policy_coef if args.policy_coef is not None else args.ranking_coef

        policy_obs = None
        policy_targets = None
        policy_weights = None
        policy_bucket_ids = None
        if policy_dataset is not None:
            policy_obs, policy_targets, policy_weights, policy_bucket_ids, policy_bucket_counts = policy_dataset
            policy_obs = policy_obs.to(device)
            policy_targets = policy_targets.to(device)
            policy_weights = policy_weights.to(device)
            policy_bucket_ids = policy_bucket_ids.to(device)

        value_obs = None
        value_targets = None
        value_weights = None
        if value_dataset is not None:
            value_obs, value_targets, value_weights, value_bucket_counts = value_dataset
            value_obs = value_obs.to(device)
            value_targets = value_targets.to(device)
            value_weights = value_weights.to(device)

        for _ in progress_iter(range(args.update_epochs), desc=f"update epoch {epoch}", unit="pass", leave=False):
            if policy_obs is not None and policy_coef > 0:
                for batch in minibatch_indices(policy_obs.shape[0], args.batch_size, rng):
                    batch_idx = torch.tensor(batch, dtype=torch.int64, device=device)
                    batch_obs = policy_obs.index_select(0, batch_idx)
                    batch_targets = policy_targets.index_select(0, batch_idx)
                    batch_weights = policy_weights.index_select(0, batch_idx)
                    batch_bucket_ids = policy_bucket_ids.index_select(0, batch_idx)
                    logits, _values = model(batch_obs, batch_bucket_ids)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    policy_loss = -(batch_targets * log_probs).sum(dim=-1)
                    policy_loss = (policy_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
                    loss = policy_coef * policy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    policy_loss_value = float(policy_loss.item())
                    update_losses.append(float(loss.item()))

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

            if policy_obs is not None and args.imitation_coef > 0:
                for batch in minibatch_indices(policy_obs.shape[0], args.batch_size, rng):
                    batch_idx = torch.tensor(batch, dtype=torch.int64, device=device)
                    batch_obs = policy_obs.index_select(0, batch_idx)
                    batch_bucket_ids = policy_bucket_ids.index_select(0, batch_idx)
                    batch_actions = torch.argmax(policy_targets.index_select(0, batch_idx), dim=-1)
                    batch_weights = policy_weights.index_select(0, batch_idx)
                    logits, _values = model(batch_obs, batch_bucket_ids)
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
        decision_source_counts = {"explore": 0, "teacher": 0}
        for decision in collected_decisions:
            decision_source = str(decision.get("source", "explore"))
            decision_source_counts[decision_source] = decision_source_counts.get(decision_source, 0) + 1
        summary = {
            "epoch": epoch,
            "sampled_case_count": len(epoch_train_cases),
            "sampled_case_counts": sampled_case_counts,
            "hard_label_case_count": hard_label_case_count,
            "train_avg_cost": train_avg_cost,
            "train_best_cost": train_best_cost,
            "train_avg_return": train_avg_return,
            "train_timeout_pct": 100.0 * train_timeout_count / max(1, len(collected)),
            "train_error_pct": 100.0 * train_error_count / max(1, len(collected)),
            "train_fail_pct": 100.0 * train_fail_count / max(1, len(collected)),
            "epsilon": epsilon,
            "loss": sum(update_losses) / max(1, len(update_losses)),
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "imitation_loss": imitation_loss_value,
            "policy_examples": 0 if policy_obs is None else int(policy_obs.shape[0]),
            "value_examples": 0 if value_obs is None else int(value_obs.shape[0]),
            "policy_bucket_counts": policy_bucket_counts,
            "value_bucket_counts": value_bucket_counts,
            "decision_source_counts": decision_source_counts,
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
                model.num_buckets,
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
                    model.num_buckets,
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
                model.num_buckets,
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
            f"policy_loss={summary['policy_loss']:.6f} "
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
        print(
            "train cases: "
            f"sampled={summary['sampled_case_count']} {summary['sampled_case_counts']} "
            f"hard_label_cases={summary['hard_label_case_count']}"
        )
        train_bucket_line = " | ".join(
            f"{bucket}:n={int(stats['count'])},ret={stats['avg_return']:.3f},fail={stats['fail_pct']:.1f}%"
            for bucket, stats in summary["train_bucket_stats"].items()
        )
        print(f"train buckets: {train_bucket_line}")
        print(
            "train datasets: "
            f"policy={summary['policy_examples']} {summary['policy_bucket_counts']} "
            f"value={summary['value_examples']} {summary['value_bucket_counts']} "
            f"sources={summary['decision_source_counts']}"
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
