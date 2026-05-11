from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .imap_env import ImapEnv, SearchState
from .model import PolicyValueNet


@dataclass
class TrajectoryStep:
    obs: np.ndarray
    action: int
    log_prob: float
    entropy: float


@dataclass
class EpisodeResult:
    steps: list[TrajectoryStep]
    final_cost: float
    final_area: int
    final_depth: int
    final_sequence: str
    final_return: float
    done_reason: str | None


@dataclass
class DecisionPoint:
    obs: np.ndarray
    action_scores: list[tuple[int, float]]


@dataclass
class BeamNode:
    node_id: int
    state: SearchState
    done: bool
    final_info: dict[str, object]
    score: float
    log_prob_sum: float
    steps: list[TrajectoryStep]


@dataclass
class PendingExpansion:
    node: BeamNode
    node_obs: np.ndarray
    probs: torch.Tensor
    entropy: float
    action_idx: int


def load_policy_checkpoint(path: Path, device: torch.device) -> tuple[PolicyValueNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyValueNet(
        checkpoint["obs_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def policy_logits_value(
    model: PolicyValueNet,
    obs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(obs_tensor)
    return logits[0].detach().cpu(), float(value.item())


def _normalized_return(env: ImapEnv, cost: float) -> float:
    if env.initial_snapshot is None:
        raise RuntimeError("Environment has not been reset.")
    init_cost = max(1e-6, float(env.initial_snapshot.cost))
    return (float(env.initial_snapshot.cost) - cost) / init_cost


def sample_episode(
    env: ImapEnv,
    model: PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    temperature: float = 1.0,
    epsilon: float = 0.1,
) -> EpisodeResult:
    obs = env.reset()
    done = False
    steps: list[TrajectoryStep] = []
    final_info: dict[str, object] | None = None

    while not done:
        logits, _value = policy_logits_value(model, obs, device)
        scaled_logits = logits / max(1e-3, temperature)
        probs = torch.softmax(scaled_logits, dim=0)

        if rng.random() < epsilon:
            action = rng.randrange(len(probs))
        else:
            action = int(torch.multinomial(probs, 1).item())

        log_prob = float(torch.log(probs[action].clamp_min(1e-12)).item())
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
        steps.append(TrajectoryStep(obs=np.array(obs, copy=True), action=action, log_prob=log_prob, entropy=entropy))
        obs, _reward, done, info = env.step(action)
        final_info = info

    if final_info is None:
        raise RuntimeError("Episode ended without final info.")

    final_cost = float(final_info["cost"])
    return EpisodeResult(
        steps=steps,
        final_cost=final_cost,
        final_area=int(final_info["area"]),
        final_depth=int(final_info["depth"]),
        final_sequence=str(final_info["sequence"]),
        final_return=_normalized_return(env, final_cost),
        done_reason=str(final_info.get("done_reason")),
    )


def _pick_action_indices(
    probs: torch.Tensor,
    branch_topk: int,
    rng: random.Random,
    gumbel_scale: float,
    random_mix_prob: float,
) -> list[int]:
    topk = min(branch_topk, probs.numel())
    if gumbel_scale > 0:
        noise = np.random.gumbel(size=probs.numel()) * gumbel_scale
        scores = probs.log().clamp_min(-30.0) + torch.tensor(noise, dtype=probs.dtype)
        top_indices = torch.topk(scores, k=topk).indices.tolist()
    else:
        top_indices = torch.topk(probs, k=topk).indices.tolist()

    if random_mix_prob > 0 and rng.random() < random_mix_prob:
        all_indices = list(range(probs.numel()))
        rng.shuffle(all_indices)
        for candidate in all_indices:
            if candidate not in top_indices:
                top_indices[-1] = candidate
                break
    return top_indices


def _simulate_branch(
    env: ImapEnv,
    state: SearchState,
    action_idx: int,
):
    return env.simulate_from_state(state, int(action_idx))


def search_candidates(
    env: ImapEnv,
    model: PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    num_candidates: int,
    beam_width: int = 4,
    branch_topk: int = 4,
    temperature: float = 1.0,
    value_weight: float = 0.3,
    prior_weight: float = 0.03,
    gumbel_scale: float = 0.0,
    random_mix_prob: float = 0.0,
    expand_workers: int = 1,
    reset_env: bool = True,
) -> tuple[list[EpisodeResult], list[DecisionPoint]]:
    if reset_env or env.current_state is None:
        env.reset()
    initial_state = env.get_state()
    initial_info = {
        "sequence": "",
        "cost": float(initial_state.snapshot.cost),
        "area": int(initial_state.snapshot.fpga.area),
        "depth": int(initial_state.snapshot.fpga.depth),
        "done_reason": None,
    }
    beam = [
        BeamNode(
            node_id=0,
            state=initial_state,
            done=False,
            final_info=initial_info,
            score=_normalized_return(env, initial_state.snapshot.cost),
            log_prob_sum=0.0,
            steps=[],
        )
    ]
    terminal_nodes: list[BeamNode] = []
    decision_points: list[DecisionPoint] = []
    node_records: dict[int, dict[str, object]] = {
        0: {
            "obs": np.array(env.observe_state(initial_state), copy=True),
            "children": [],
            "action": None,
            "done": False,
            "terminal_return": None,
            "bootstrap_return": _normalized_return(env, initial_state.snapshot.cost),
        }
    }
    next_node_id = 1

    for _ in range(env.max_steps):
        expanded: list[BeamNode] = []
        pending: list[PendingExpansion] = []
        for node in beam:
            if node.done:
                terminal_nodes.append(node)
                continue

            node_obs = env.observe_state(node.state)
            logits, _value = policy_logits_value(model, node_obs, device)
            probs = torch.softmax(logits / max(1e-3, temperature), dim=0)
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
            chosen_indices = _pick_action_indices(
                probs=probs,
                branch_topk=branch_topk,
                rng=rng,
                gumbel_scale=gumbel_scale,
                random_mix_prob=random_mix_prob,
            )

            for action_idx in chosen_indices:
                pending.append(
                    PendingExpansion(
                        node=node,
                        node_obs=np.array(node_obs, copy=True),
                        probs=probs,
                        entropy=entropy,
                        action_idx=int(action_idx),
                    )
                )

        if expand_workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=expand_workers) as executor:
                futures = {
                    executor.submit(_simulate_branch, env, item.node.state, item.action_idx): item
                    for item in pending
                }
                results = []
                for future in as_completed(futures):
                    results.append((futures[future], future.result()))
        else:
            results = [
                (item, _simulate_branch(env, item.node.state, item.action_idx))
                for item in pending
            ]

        for item, (next_state, _reward, done, info) in results:
            prob = float(item.probs[item.action_idx].item())
            next_obs = env.observe_state(next_state)
            if done:
                predicted = 0.0
                rank_return = _normalized_return(env, float(info["cost"]))
            else:
                _next_logits, next_value = policy_logits_value(model, next_obs, device)
                predicted = next_value
                rank_return = _normalized_return(env, next_state.snapshot.cost)
            child_node_id = next_node_id
            next_node_id += 1
            bootstrap_return = rank_return + value_weight * predicted
            node_records[child_node_id] = {
                "obs": np.array(next_obs, copy=True),
                "children": [],
                "action": int(item.action_idx),
                "parent": item.node.node_id,
                "done": bool(done),
                "terminal_return": rank_return if done else None,
                "bootstrap_return": bootstrap_return,
            }
            node_records[item.node.node_id]["children"].append(child_node_id)

            step = TrajectoryStep(
                obs=item.node_obs,
                action=int(item.action_idx),
                log_prob=float(torch.log(item.probs[item.action_idx].clamp_min(1e-12)).item()),
                entropy=item.entropy,
            )
            new_log_prob = item.node.log_prob_sum + math.log(max(prob, 1e-12))
            score = bootstrap_return + prior_weight * new_log_prob
            child = BeamNode(
                node_id=child_node_id,
                state=next_state,
                done=done,
                final_info=info,
                score=score,
                log_prob_sum=new_log_prob,
                steps=item.node.steps + [step],
            )
            if done:
                terminal_nodes.append(child)
            else:
                expanded.append(child)

        expanded.sort(key=lambda item: item.score, reverse=True)
        beam = expanded[:beam_width]
        if len(terminal_nodes) >= num_candidates and not beam:
            break
        if not beam:
            break

    best_return: dict[int, float] = {}
    for node_id in sorted(node_records.keys(), reverse=True):
        record = node_records[node_id]
        base_return = record["terminal_return"]
        if base_return is None:
            base_return = float(record["bootstrap_return"])
        child_best = base_return
        for child_id in record["children"]:
            child_best = max(child_best, best_return.get(child_id, float(node_records[child_id]["bootstrap_return"])))
        best_return[node_id] = float(child_best)

    decision_points = []
    for node_id, record in node_records.items():
        children = record["children"]
        if len(children) < 2:
            continue
        grouped_scores: dict[int, list[float]] = {}
        for child_id in children:
            child_record = node_records[child_id]
            action_idx = int(child_record["action"])
            child_score = best_return.get(child_id, float(child_record["bootstrap_return"]))
            grouped_scores.setdefault(action_idx, []).append(float(child_score))
        if len(grouped_scores) < 2:
            continue
        decision_points.append(
            DecisionPoint(
                obs=np.array(record["obs"], copy=True),
                action_scores=[(action_idx, max(scores)) for action_idx, scores in grouped_scores.items()],
            )
        )

    if not terminal_nodes:
        terminal_nodes = [min(beam, key=lambda item: float(item.final_info["cost"]))] if beam else []

    terminal_nodes.sort(key=lambda item: float(item.final_info["cost"]))
    results = []
    for node in terminal_nodes[:num_candidates]:
        final_cost = float(node.final_info["cost"])
        results.append(
            EpisodeResult(
                steps=node.steps,
                final_cost=final_cost,
                final_area=int(node.final_info["area"]),
                final_depth=int(node.final_info["depth"]),
                final_sequence=str(node.final_info["sequence"]),
                final_return=_normalized_return(env, final_cost),
                done_reason=str(node.final_info.get("done_reason")),
            )
        )
    return results, decision_points


def beam_search(
    env: ImapEnv,
    model: PolicyValueNet,
    device: torch.device,
    beam_width: int = 4,
    branch_topk: int = 4,
    temperature: float = 1.0,
    value_weight: float = 0.3,
    prior_weight: float = 0.03,
    expand_workers: int = 1,
    reset_env: bool = True,
) -> dict[str, object]:
    if reset_env or env.current_state is None:
        env.reset()
    initial_state = env.get_state()
    initial_info = {
        "sequence": "",
        "cost": float(initial_state.snapshot.cost),
        "area": int(initial_state.snapshot.fpga.area),
        "depth": int(initial_state.snapshot.fpga.depth),
        "done_reason": None,
    }
    beam = [
        BeamNode(
            node_id=-1,
            state=initial_state,
            done=False,
            final_info=initial_info,
            score=_normalized_return(env, initial_state.snapshot.cost),
            log_prob_sum=0.0,
            steps=[],
        )
    ]
    best_done: BeamNode | None = None

    for _ in range(env.max_steps):
        expanded: list[BeamNode] = []
        pending: list[PendingExpansion] = []
        for node in beam:
            if node.done:
                expanded.append(node)
                continue

            node_obs = env.observe_state(node.state)
            logits, _value = policy_logits_value(model, node_obs, device)
            probs = torch.softmax(logits / max(1e-3, temperature), dim=0)
            topk = min(branch_topk, probs.numel())
            top_probs, top_indices = torch.topk(probs, k=topk)

            for prob, action_idx in zip(top_probs.tolist(), top_indices.tolist()):
                pending.append(
                    PendingExpansion(
                        node=node,
                        node_obs=np.array(node_obs, copy=True),
                        probs=probs,
                        entropy=0.0,
                        action_idx=int(action_idx),
                    )
                )

        if expand_workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=expand_workers) as executor:
                futures = {
                    executor.submit(_simulate_branch, env, item.node.state, item.action_idx): item
                    for item in pending
                }
                results = []
                for future in as_completed(futures):
                    results.append((futures[future], future.result()))
        else:
            results = [
                (item, _simulate_branch(env, item.node.state, item.action_idx))
                for item in pending
            ]

        for item, (next_state, _reward, done, info) in results:
            prob = float(item.probs[item.action_idx].item())
            next_obs = env.observe_state(next_state)
            if done:
                predicted = 0.0
                rank_return = _normalized_return(env, float(info["cost"]))
            else:
                _next_logits, next_value = policy_logits_value(model, next_obs, device)
                predicted = next_value
                rank_return = _normalized_return(env, next_state.snapshot.cost)

            new_log_prob = item.node.log_prob_sum + math.log(max(prob, 1e-12))
            score = rank_return + value_weight * predicted + prior_weight * new_log_prob
            child = BeamNode(
                node_id=-1,
                state=next_state,
                done=done,
                final_info=info,
                score=score,
                log_prob_sum=new_log_prob,
                steps=item.node.steps,
            )
            expanded.append(child)
            if done and (
                best_done is None or float(info["cost"]) < float(best_done.final_info["cost"])
            ):
                best_done = child

        expanded.sort(key=lambda item: item.score, reverse=True)
        beam = expanded[:beam_width]
        if beam and all(node.done for node in beam):
            break

    winner = best_done
    if winner is None:
        winner = min(beam, key=lambda item: float(item.final_info["cost"]))

    return {
        "sequence": str(winner.final_info["sequence"]),
        "cost": float(winner.final_info["cost"]),
        "area": int(winner.final_info["area"]),
        "depth": int(winner.final_info["depth"]),
        "done_reason": winner.final_info.get("done_reason"),
    }
