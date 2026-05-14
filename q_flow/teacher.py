from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import subprocess
import time

import numpy as np

from .actions import FlowAction
from .env import AIGEnv, FinalStats, NetStats, StateStats


@dataclass
class TeacherRecord:
    obs: np.ndarray
    best_action: int
    q_values: np.ndarray
    action_mask: np.ndarray
    target_value: float
    target_area: float
    target_depth: float

    def to_payload(self) -> dict[str, np.ndarray | float | int]:
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "best_action": int(self.best_action),
            "q_values": np.asarray(self.q_values, dtype=np.float32),
            "action_mask": np.asarray(self.action_mask, dtype=np.float32),
            "target_value": float(self.target_value),
            "target_area": float(self.target_area),
            "target_depth": float(self.target_depth),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, np.ndarray | float | int]) -> "TeacherRecord":
        return cls(
            obs=np.asarray(payload["obs"], dtype=np.float32),
            best_action=int(payload["best_action"]),
            q_values=np.asarray(payload["q_values"], dtype=np.float32),
            action_mask=np.asarray(payload["action_mask"], dtype=np.float32),
            target_value=float(payload["target_value"]),
            target_area=float(payload["target_area"]),
            target_depth=float(payload["target_depth"]),
        )


@dataclass
class SearchTarget:
    best_action: int
    best_return: float
    best_final: FinalStats
    q_values: np.ndarray
    action_mask: np.ndarray


def estimate_teacher_states(
    max_steps: int,
    branch_topk: int,
    deep_branch_topk: int = 2,
    tail_branch_topk: int = 1,
) -> int:
    total = 0
    width = 1
    for depth in range(max_steps):
        total += width
        if depth == 0:
            width *= max(1, branch_topk)
        elif depth == 1:
            width *= max(1, deep_branch_topk)
        else:
            width *= max(1, tail_branch_topk)
    return max(1, total)


def _normalized_return(initial_final: FinalStats, final_stats: FinalStats) -> float:
    return (initial_final.cost - final_stats.cost) / max(1e-6, initial_final.cost)


def _proxy_cost(stats: NetStats) -> float:
    return (
        0.55 * np.log1p(float(stats.depth))
        + 0.35 * np.log1p(float(stats.area))
        + 0.06 * float(stats.high_level_ratio)
        + 0.04 * float(stats.high_fanout_ratio)
    )


def _rank_nonterminal_actions(
    env: AIGEnv,
    state: StateStats,
    action_indices: list[int],
    branch_topk: int,
) -> list[int]:
    if len(action_indices) <= branch_topk:
        return action_indices

    ranked: list[tuple[float, int]] = []
    state_proxy = _proxy_cost(state.aig)
    for action_index in action_indices:
        try:
            next_state = env.next_state(state, action_index)
        except Exception:
            continue
        repeat_penalty = 0.03 * float(state.action_counts[action_index])
        score = state_proxy - _proxy_cost(next_state.aig) - repeat_penalty
        ranked.append((score, action_index))
    ranked.sort(reverse=True)
    return [action_index for _score, action_index in ranked[:branch_topk]]


def _best_terminal_for_state(
    env: AIGEnv,
    state: StateStats,
    terminal_indices: list[int],
) -> tuple[int, FinalStats]:
    best_index = terminal_indices[0]
    best_final = env.evaluate_final(
        state.sequence,
        env.actions[best_index].final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1",
    )
    best_cost = best_final.cost
    for action_index in terminal_indices[1:]:
        action = env.actions[action_index]
        final_stats = env.evaluate_final(state.sequence, action.final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1")
        if final_stats.cost < best_cost:
            best_index = action_index
            best_final = final_stats
            best_cost = final_stats.cost
    return best_index, best_final


def _branch_limit(
    step_index: int,
    branch_topk: int,
    deep_branch_topk: int,
    tail_branch_topk: int,
) -> int:
    if step_index <= 0:
        return max(1, branch_topk)
    if step_index == 1:
        return max(1, min(branch_topk, deep_branch_topk))
    return max(1, min(branch_topk, tail_branch_topk))


def _terminal_limit(
    step_index: int,
    terminal_topk: int,
    deep_terminal_topk: int,
) -> int:
    if step_index <= 0:
        return max(1, terminal_topk)
    return max(1, min(terminal_topk, deep_terminal_topk))


def build_teacher_records(
    env: AIGEnv,
    actions: list[FlowAction],
    max_steps: int,
    branch_topk: int = 4,
    deep_branch_topk: int = 2,
    tail_branch_topk: int = 1,
    terminal_topk: int = 2,
    deep_terminal_topk: int = 1,
    teacher_budget_sec: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[TeacherRecord]:
    obs0 = env.reset()
    if obs0 is None or env.initial_state is None:
        raise RuntimeError("env.reset failed")
    initial_state = env.initial_state
    initial_final = env.evaluate_final(tuple(), "map_fpga -P 10 -C 6 -G 1 -L 1")
    terminal_indices = [idx for idx, action in enumerate(actions) if action.terminal]
    if not terminal_indices:
        raise RuntimeError("no terminal actions configured")
    teacher_terminal_indices = sorted(
        terminal_indices,
        key=lambda idx: actions[idx].teacher_priority,
        reverse=True,
    )[: max(1, terminal_topk)]
    fast_terminal_index = next(
        (
            idx
            for idx in terminal_indices
            if (actions[idx].final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1") == "map_fpga -P 10 -C 6 -G 1 -L 1"
        ),
        teacher_terminal_indices[0],
    )
    deadline = None if teacher_budget_sec is None or teacher_budget_sec <= 0.0 else (time.monotonic() + teacher_budget_sec)

    memo: dict[tuple[tuple[str, ...], int], SearchTarget] = {}
    records: dict[tuple[tuple[str, ...], int], TeacherRecord] = {}

    def safe_final(sequence: tuple[str, ...], map_command: str) -> FinalStats | None:
        try:
            return env.evaluate_final(sequence, map_command)
        except (subprocess.TimeoutExpired, RuntimeError):
            return None

    def safe_next(state: StateStats, action_index: int) -> StateStats | None:
        try:
            return env.next_state(state, action_index)
        except (subprocess.TimeoutExpired, RuntimeError):
            return None

    def finalize_state(
        state: StateStats,
        key: tuple[tuple[str, ...], int],
        best_action: int,
        best_return: float,
        best_final: FinalStats,
        q_values: np.ndarray,
        action_mask: np.ndarray,
    ) -> SearchTarget:
        record = TeacherRecord(
            obs=np.array(env.observe(state), copy=True),
            best_action=best_action,
            q_values=q_values,
            action_mask=action_mask,
            target_value=best_return,
            target_area=float(np.log1p(best_final.area)),
            target_depth=float(np.log1p(best_final.depth)),
        )
        records[key] = record
        target = SearchTarget(
            best_action=best_action,
            best_return=best_return,
            best_final=best_final,
            q_values=q_values,
            action_mask=action_mask,
        )
        memo[key] = target
        return target

    def fallback_target(
        state: StateStats,
        key: tuple[tuple[str, ...], int],
        q_values: np.ndarray,
        action_mask: np.ndarray,
    ) -> SearchTarget:
        fallback_indices = [fast_terminal_index, *teacher_terminal_indices]
        tried: set[int] = set()
        for idx in fallback_indices:
            if idx in tried:
                continue
            tried.add(idx)
            final_stats = safe_final(state.sequence, actions[idx].final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1")
            if final_stats is None:
                continue
            best_return = _normalized_return(initial_final, final_stats)
            q_values[idx] = float(best_return)
            action_mask[idx] = 1.0
            return finalize_state(state, key, idx, best_return, final_stats, q_values, action_mask)
        raise RuntimeError("teacher search failed to produce any final state")

    def select_terminal_indices(state: StateStats) -> list[int]:
        limit = _terminal_limit(state.step_index, terminal_topk, deep_terminal_topk)
        if state.step_index <= 0:
            return teacher_terminal_indices[:limit]
        chosen = [fast_terminal_index]
        for idx in teacher_terminal_indices:
            if idx == fast_terminal_index:
                continue
            chosen.append(idx)
            if len(chosen) >= limit:
                break
        return chosen[:limit]

    def solve(state: StateStats) -> SearchTarget:
        key = (state.sequence, state.step_index)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if progress_callback is not None:
            progress_callback(1)

        q_values = np.zeros(len(actions), dtype=np.float32)
        action_mask = np.zeros(len(actions), dtype=np.float32)
        best_action = terminal_indices[0]
        best_final: FinalStats | None = None
        best_return = -1e9
        if deadline is not None and time.monotonic() >= deadline:
            return fallback_target(state, key, q_values, action_mask)

        nonterminal_indices = [idx for idx, action in enumerate(actions) if not action.terminal]
        state_terminal_indices = select_terminal_indices(state)
        if state.step_index + 1 < max_steps:
            candidate_nonterminal = _rank_nonterminal_actions(
                env,
                state,
                nonterminal_indices,
                _branch_limit(state.step_index, branch_topk, deep_branch_topk, tail_branch_topk),
            )
        else:
            candidate_nonterminal = nonterminal_indices
        candidate_indices = list(state_terminal_indices) + candidate_nonterminal

        for action_index in candidate_indices:
            if deadline is not None and time.monotonic() >= deadline and best_final is not None:
                break
            action = actions[action_index]
            action_mask[action_index] = 1.0

            if action.terminal:
                final_stats = safe_final(
                    state.sequence,
                    action.final_map_command or "map_fpga -P 10 -C 6 -G 1 -L 1",
                )
                if final_stats is None:
                    continue
                action_return = _normalized_return(initial_final, final_stats)
            else:
                next_state = safe_next(state, action_index)
                if next_state is None:
                    continue
                if next_state.step_index >= max_steps:
                    try:
                        _best_terminal_idx, final_stats = _best_terminal_for_state(env, next_state, state_terminal_indices)
                    except (subprocess.TimeoutExpired, RuntimeError):
                        continue
                    action_return = _normalized_return(initial_final, final_stats)
                else:
                    try:
                        child_target = solve(next_state)
                    except RuntimeError:
                        continue
                    final_stats = child_target.best_final
                    action_return = child_target.best_return

            q_values[action_index] = float(action_return)
            if action_return > best_return:
                best_return = float(action_return)
                best_action = action_index
                best_final = final_stats

        if best_final is None:
            return fallback_target(state, key, q_values, action_mask)

        return finalize_state(state, key, best_action, best_return, best_final, q_values, action_mask)

    solve(initial_state)
    ordered_keys = sorted(records.keys(), key=lambda item: (item[1], item[0]))
    return [records[key] for key in ordered_keys]
