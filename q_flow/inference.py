from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .actions import FlowAction
from .env import AIGEnv, FinalStats, StateStats
from .model import GreedyQNet


@dataclass
class InferenceResult:
    action_indices: list[int]
    command_sequence: list[str]
    final_map_command: str
    final_stats: FinalStats


def load_checkpoint(path, device: torch.device) -> GreedyQNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = GreedyQNet(checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _predict_q(model: GreedyQNet, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values, _value, _area, _depth = model(obs_tensor)
    return q_values[0].detach().cpu().numpy()


def _top_margin(q_values: np.ndarray, legal_indices: list[int]) -> float:
    if len(legal_indices) <= 1:
        return 1e9
    legal_scores = sorted((float(q_values[idx]), idx) for idx in legal_indices)
    return legal_scores[-1][0] - legal_scores[-2][0]


def _best_terminal_for_state(
    env: AIGEnv,
    state: StateStats,
    actions: list[FlowAction],
    legal_terminal_indices: list[int],
) -> tuple[int, FinalStats]:
    best_index = legal_terminal_indices[0]
    best_final = env.evaluate_final(state.sequence, actions[best_index].final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1")
    best_cost = best_final.cost
    for action_index in legal_terminal_indices[1:]:
        action = actions[action_index]
        final_stats = env.evaluate_final(state.sequence, action.final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1")
        if final_stats.cost < best_cost:
            best_index = action_index
            best_final = final_stats
            best_cost = final_stats.cost
    return best_index, best_final


def _local_rollout_cost(
    env: AIGEnv,
    state: StateStats,
    first_action_index: int,
    actions: list[FlowAction],
    model: GreedyQNet,
    device: torch.device,
    local_depth: int,
    local_topk: int,
) -> tuple[float, list[int], FinalStats]:
    action = actions[first_action_index]
    if action.terminal:
        final_stats = env.evaluate_final(state.sequence, action.final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1")
        return final_stats.cost, [first_action_index], final_stats

    next_state = env.next_state(state, first_action_index)
    rollout = [first_action_index]
    if local_depth <= 1 or next_state.step_index >= env.max_steps:
        terminal_indices = [idx for idx, item in enumerate(actions) if item.terminal]
        best_terminal_idx, best_final = _best_terminal_for_state(env, next_state, actions, terminal_indices)
        rollout.append(best_terminal_idx)
        return best_final.cost, rollout, best_final

    next_q_values = _predict_q(model, env.observe(next_state), device)
    next_q = np.argsort(next_q_values)[::-1]
    candidate_indices: list[int] = []
    for idx in next_q:
        idx = int(idx)
        candidate_indices.append(idx)
        if len(candidate_indices) >= local_topk:
            break
    terminal_indices = [idx for idx, item in enumerate(actions) if item.terminal]
    for terminal_index in terminal_indices:
        if terminal_index not in candidate_indices:
            candidate_indices.append(terminal_index)

    best_cost = float("inf")
    best_path: list[int] | None = None
    best_final: FinalStats | None = None
    for second_idx in candidate_indices:
        second_action = actions[second_idx]
        if second_action.terminal:
            final_stats = env.evaluate_final(next_state.sequence, second_action.final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1")
            cost = final_stats.cost
            path = rollout + [second_idx]
        else:
            second_state = env.next_state(next_state, second_idx)
            terminal_idx, final_stats = _best_terminal_for_state(env, second_state, actions, terminal_indices)
            cost = final_stats.cost
            path = rollout + [second_idx, terminal_idx]
        if cost < best_cost:
            best_cost = cost
            best_path = path
            best_final = final_stats
    if best_path is None or best_final is None:
        raise RuntimeError("local rollout did not produce a path")
    return best_cost, best_path, best_final


def run_policy(
    env: AIGEnv,
    actions: list[FlowAction],
    model: GreedyQNet,
    device: torch.device,
    confidence_margin: float = 0.08,
    fallback_topk: int = 3,
    fallback_depth: int = 2,
) -> InferenceResult:
    obs = env.reset()
    if env.initial_state is None:
        raise RuntimeError("env reset failed")
    state = env.initial_state
    chosen_action_indices: list[int] = []
    chosen_commands: list[str] = []
    final_map_command = "map_fpga -P 10 -C 6 -G 1 -L 1"

    while state.step_index < env.max_steps:
        q_values = _predict_q(model, obs, device)
        legal_indices = list(range(len(actions)))
        margin = _top_margin(q_values, legal_indices)
        ranked_indices = [int(idx) for idx in np.argsort(q_values)[::-1]]
        action_index = ranked_indices[0]

        if margin < confidence_margin:
            candidate_indices = ranked_indices[: max(1, fallback_topk)]
            terminal_indices = [idx for idx, action in enumerate(actions) if action.terminal]
            for terminal_index in terminal_indices:
                if terminal_index not in candidate_indices:
                    candidate_indices.append(terminal_index)
            best_cost = float("inf")
            best_first = action_index
            for candidate_index in candidate_indices:
                cost, _path, _final = _local_rollout_cost(
                    env=env,
                    state=state,
                    first_action_index=candidate_index,
                    actions=actions,
                    model=model,
                    device=device,
                    local_depth=fallback_depth,
                    local_topk=fallback_topk,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_first = candidate_index
            action_index = best_first

        action = actions[action_index]
        chosen_action_indices.append(action_index)
        if action.terminal:
            final_map_command = action.final_map_command or final_map_command
            break
        chosen_commands.extend(action.commands)
        state = env.next_state(state, action_index)
        obs = env.observe(state)
    else:
        terminal_indices = [idx for idx, action in enumerate(actions) if action.terminal]
        best_terminal_idx, _ = _best_terminal_for_state(env, state, actions, terminal_indices)
        chosen_action_indices.append(best_terminal_idx)
        final_map_command = actions[best_terminal_idx].final_map_command or final_map_command

    final_stats = env.evaluate_final(state.sequence, final_map_command)
    command_sequence = [*chosen_commands, final_map_command]
    return InferenceResult(
        action_indices=chosen_action_indices,
        command_sequence=command_sequence,
        final_map_command=final_map_command,
        final_stats=final_stats,
    )
