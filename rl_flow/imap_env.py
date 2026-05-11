from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import MacroAction


AIG_RE = re.compile(r"Stats of AIG: pis=(\d+), pos=(\d+), area=(\d+), depth=(\d+)")
FPGA_RE = re.compile(r"Stats of FPGA: pis=(\d+), pos=(\d+), area=(\d+), depth=(\d+)")


@dataclass
class NetStats:
    pis: int
    pos: int
    area: int
    depth: int


@dataclass
class EvalSnapshot:
    aig: NetStats
    fpga: NetStats
    sequence: tuple[str, ...]

    @property
    def cost(self) -> float:
        return 0.6 * self.fpga.depth + 0.4 * self.fpga.area


@dataclass
class SearchState:
    sequence: tuple[str, ...]
    snapshot: EvalSnapshot
    step_index: int
    last_action_index: int
    action_counts: np.ndarray


class ImapEnv:
    _EVAL_CACHE: dict[tuple[str, tuple[str, ...], str], EvalSnapshot] = {}
    _CACHE_MAX_ENTRIES = 200000

    def __init__(
        self,
        input_aig: Path,
        imap_bin: Path,
        actions: list[MacroAction],
        max_steps: int = 4,
        probe_map_command: str = "map_fpga -P 12 -C 6 -G 1 -L 2",
        timeout_sec: float | None = 60.0,
    ) -> None:
        self.input_aig = Path(input_aig).resolve()
        self.imap_bin = Path(imap_bin).resolve()
        self.actions = actions
        self.max_steps = max_steps
        self.probe_map_command = probe_map_command
        self.timeout_sec = timeout_sec
        self.initial_snapshot: EvalSnapshot | None = None
        self.current_state: SearchState | None = None

    def reset(self) -> np.ndarray:
        initial_snapshot = self._evaluate(tuple(), self.probe_map_command)
        self.initial_snapshot = initial_snapshot
        self.current_state = SearchState(
            sequence=tuple(),
            snapshot=initial_snapshot,
            step_index=0,
            last_action_index=-1,
            action_counts=np.zeros(len(self.actions), dtype=np.float32),
        )
        return self.observe_state(self.current_state)

    def step(self, action_index: int):
        if self.current_state is None:
            raise RuntimeError("Call reset() before step().")
        next_state, reward, done, info = self.simulate_from_state(self.current_state, action_index)
        self.current_state = next_state
        return self.observe_state(next_state), reward, done, info

    def step_with_probe(self, action_index: int, probe: dict):
        if self.current_state is None:
            raise RuntimeError("Call reset() before step_with_probe().")
        next_state, reward, done, info = self.transition_from_probe(self.current_state, action_index, probe)
        self.current_state = next_state
        return self.observe_state(next_state), reward, done, info

    def evaluate_action(self, action_index: int) -> dict:
        if self.current_state is None:
            raise RuntimeError("Call reset() before evaluate_action().")
        transition = self._simulate_action_from_state(self.current_state, action_index)
        next_state, reward, done, info = self.transition_from_probe(
            self.current_state,
            action_index,
            transition,
        )
        return {
            "transition": transition,
            "state": next_state,
            "obs": self.observe_state(next_state),
            "reward": reward,
            "done": done,
            "info": info,
            "snapshot": next_state.snapshot,
        }

    def get_state(self) -> SearchState:
        if self.current_state is None:
            raise RuntimeError("Call reset() before get_state().")
        return SearchState(
            sequence=self.current_state.sequence,
            snapshot=self.current_state.snapshot,
            step_index=self.current_state.step_index,
            last_action_index=self.current_state.last_action_index,
            action_counts=self.current_state.action_counts.copy(),
        )

    def observe_state(self, state: SearchState) -> np.ndarray:
        return self._encode_state(state)

    def simulate_from_state(self, state: SearchState, action_index: int):
        transition = self._simulate_action_from_state(state, action_index)
        return self.transition_from_probe(state, action_index, transition)

    def transition_from_probe(self, state: SearchState, action_index: int, probe: dict):
        next_state = SearchState(
            sequence=probe["next_sequence"],
            snapshot=probe["next_snapshot"],
            step_index=state.step_index + 1,
            last_action_index=action_index,
            action_counts=self._next_action_counts(state, action_index),
        )
        info = {
            "done_reason": probe["done_reason"],
            "sequence": probe["sequence_str"],
            "cost": probe["next_snapshot"].cost,
            "area": probe["next_snapshot"].fpga.area,
            "depth": probe["next_snapshot"].fpga.depth,
        }
        return next_state, probe["reward"], probe["done"], info

    def current_seq(self, final_map_command: str | None = None) -> str:
        if self.current_state is None:
            raise RuntimeError("Call reset() before current_seq().")
        commands = list(self.current_state.sequence)
        if final_map_command is not None:
            commands.append(final_map_command)
        return "; ".join(commands) + ";"

    def _next_action_counts(self, state: SearchState, action_index: int) -> np.ndarray:
        counts = state.action_counts.copy()
        counts[action_index] += 1.0
        return counts

    def _normalized_reward(self, raw_delta: float, terminal: bool) -> float:
        if self.initial_snapshot is None:
            raise RuntimeError("Environment has not been reset.")
        scale = max(1.0, float(self.initial_snapshot.cost))
        reward = raw_delta / scale
        if not terminal:
            reward -= 0.002
        reward = max(-2.0, min(2.0, reward))
        return reward

    def _failed_transition(
        self,
        state: SearchState,
        action: MacroAction,
        reason: str,
    ) -> dict:
        prev_snapshot = state.snapshot
        fallback_map = self.probe_map_command
        fallback_sequence = list(state.sequence)
        attempted_sequence = fallback_sequence + list(action.commands)
        reward = -1.0
        sequence_str = "; ".join(fallback_sequence + [fallback_map]) + ";"
        return {
            "next_sequence": tuple(state.sequence),
            "next_snapshot": prev_snapshot,
            "reward": reward,
            "done": True,
            "done_reason": reason,
            "sequence_str": sequence_str,
            "attempted_sequence_str": "; ".join(attempted_sequence) + ";" if attempted_sequence else "",
        }

    def _simulate_action_from_state(self, state: SearchState, action_index: int) -> dict:
        action = self.actions[action_index]
        prev_snapshot = state.snapshot
        next_sequence = list(state.sequence)

        try:
            if action.terminal or state.step_index + 1 >= self.max_steps:
                final_map = action.final_map_command or self.probe_map_command
                final_snapshot = self._evaluate(tuple(next_sequence), final_map)
                reward = self._normalized_reward(prev_snapshot.cost - final_snapshot.cost, terminal=True)
                sequence_str = "; ".join(next_sequence + [final_map]) + ";"
                return {
                    "next_sequence": tuple(next_sequence),
                    "next_snapshot": final_snapshot,
                    "reward": reward,
                    "done": True,
                    "done_reason": "terminal_action" if action.terminal else "max_steps",
                    "sequence_str": sequence_str,
                }

            next_sequence.extend(action.commands)
            next_snapshot = self._evaluate(tuple(next_sequence), self.probe_map_command)
            reward = self._normalized_reward(prev_snapshot.cost - next_snapshot.cost, terminal=False)
            sequence_str = "; ".join(next_sequence) + ";"
            return {
                "next_sequence": tuple(next_sequence),
                "next_snapshot": next_snapshot,
                "reward": reward,
                "done": False,
                "done_reason": None,
                "sequence_str": sequence_str,
            }
        except (subprocess.TimeoutExpired, RuntimeError):
            return self._failed_transition(
                state=state,
                action=action,
                reason="action_failed",
            )

    def _evaluate(self, sequence: tuple[str, ...], map_command: str) -> EvalSnapshot:
        if not self.input_aig.is_file():
            raise FileNotFoundError(self.input_aig)
        if not self.imap_bin.is_file():
            raise FileNotFoundError(self.imap_bin)

        cache_key = (str(self.input_aig), sequence, map_command)
        cached = self._EVAL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        commands = [f"read_aiger -f {self.input_aig}"]
        commands.extend(sequence)
        commands.append("print_stats -t 0")
        commands.append(map_command)
        commands.append("print_stats -t 1")
        script = "; ".join(commands) + ";"

        proc = subprocess.run(
            [str(self.imap_bin), "-c", script],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)

        aig_match = AIG_RE.search(proc.stdout)
        fpga_match = FPGA_RE.search(proc.stdout)
        if aig_match is None or fpga_match is None:
            raise RuntimeError(f"Failed to parse stats from iMAP output:\n{proc.stdout}")

        aig_stats = NetStats(*(int(x) for x in aig_match.groups()))
        fpga_stats = NetStats(*(int(x) for x in fpga_match.groups()))
        snapshot = EvalSnapshot(aig=aig_stats, fpga=fpga_stats, sequence=sequence)
        if len(self._EVAL_CACHE) >= self._CACHE_MAX_ENTRIES:
            self._EVAL_CACHE.clear()
        self._EVAL_CACHE[cache_key] = snapshot
        return snapshot

    def _encode_state(
        self,
        state: SearchState,
    ) -> np.ndarray:
        if self.initial_snapshot is None:
            raise RuntimeError("Environment has not been reset.")

        snapshot = state.snapshot
        init = self.initial_snapshot

        base = np.array(
            [
                float(snapshot.aig.pis),
                float(snapshot.aig.pos),
                float(snapshot.aig.area),
                float(snapshot.aig.depth),
                float(snapshot.fpga.area),
                float(snapshot.fpga.depth),
                float(snapshot.cost),
                float(init.aig.area),
                float(init.aig.depth),
                float(init.fpga.area),
                float(init.fpga.depth),
                float(init.cost),
                float(snapshot.aig.area) / max(1.0, float(init.aig.area)),
                float(snapshot.aig.depth) / max(1.0, float(init.aig.depth)),
                float(snapshot.fpga.area) / max(1.0, float(init.fpga.area)),
                float(snapshot.fpga.depth) / max(1.0, float(init.fpga.depth)),
                float(init.cost - snapshot.cost),
                float(snapshot.fpga.area - init.fpga.area),
                float(snapshot.fpga.depth - init.fpga.depth),
                float(state.step_index),
                float(self.max_steps - state.step_index),
                float(state.last_action_index),
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, state.action_counts.astype(np.float32)], axis=0)
