from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import MacroAction


@dataclass
class NetStats:
    pis: int
    pos: int
    area: int
    depth: int
    inv: int = 0
    po_inv: int = 0
    fanout_sum: int = 0
    fanout_max: int = 0
    fanout_ge4: int = 0
    level_sum: int = 0
    level_ge_half: int = 0
    lut1: int = 0
    lut2: int = 0
    lut3: int = 0
    lut4: int = 0
    lut5: int = 0
    lut6: int = 0

    @property
    def fanout_avg(self) -> float:
        return float(self.fanout_sum) / max(1.0, float(self.area))

    @property
    def level_avg(self) -> float:
        return float(self.level_sum) / max(1.0, float(self.area))

    @property
    def high_fanout_ratio(self) -> float:
        return float(self.fanout_ge4) / max(1.0, float(self.area))

    @property
    def high_level_ratio(self) -> float:
        return float(self.level_ge_half) / max(1.0, float(self.area))

    @property
    def inv_ratio(self) -> float:
        return float(self.inv) / max(1.0, float(self.area) * 2.0)

    @property
    def po_inv_ratio(self) -> float:
        return float(self.po_inv) / max(1.0, float(self.pos))

    @property
    def lut_hist(self) -> tuple[int, int, int, int, int, int]:
        return (self.lut1, self.lut2, self.lut3, self.lut4, self.lut5, self.lut6)


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
    _EVAL_CACHE: dict[tuple[str, tuple[str, ...], str, bool], EvalSnapshot] = {}
    _CACHE_MAX_ENTRIES = 200000

    def __init__(
        self,
        input_aig: Path,
        imap_bin: Path,
        actions: list[MacroAction],
        max_steps: int = 4,
        probe_map_command: str = "map_fpga -P 12 -C 6 -G 1 -L 2",
        timeout_sec: float | None = 60.0,
        use_history: bool = True,
        history_capacity: int = 5,
    ) -> None:
        self.input_aig = Path(input_aig).resolve()
        self.imap_bin = Path(imap_bin).resolve()
        self.actions = actions
        self.max_steps = max_steps
        self.probe_map_command = probe_map_command
        self.timeout_sec = timeout_sec
        self.use_history = use_history
        self.history_capacity = history_capacity
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
        return self._sequence_str(self.current_state.sequence, final_map_command)

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
        sequence_str = self._sequence_str(tuple(fallback_sequence), fallback_map)
        return {
            "next_sequence": tuple(state.sequence),
            "next_snapshot": prev_snapshot,
            "reward": reward,
            "done": True,
            "done_reason": reason,
            "sequence_str": sequence_str,
            "attempted_sequence_str": self._sequence_str(tuple(attempted_sequence), None) if attempted_sequence else "",
        }

    def _simulate_action_from_state(self, state: SearchState, action_index: int) -> dict:
        action = self.actions[action_index]
        prev_snapshot = state.snapshot
        next_sequence = list(state.sequence)

        try:
            if action.terminal:
                final_map = action.final_map_command or self.probe_map_command
                final_snapshot = self._evaluate(tuple(next_sequence), final_map)
                reward = self._normalized_reward(prev_snapshot.cost - final_snapshot.cost, terminal=True)
                sequence_str = self._sequence_str(tuple(next_sequence), final_map)
                return {
                    "next_sequence": tuple(next_sequence),
                    "next_snapshot": final_snapshot,
                    "reward": reward,
                    "done": True,
                    "done_reason": "terminal_action",
                    "sequence_str": sequence_str,
                }

            next_sequence.extend(action.commands)
            if state.step_index + 1 >= self.max_steps:
                final_snapshot = self._evaluate(tuple(next_sequence), self.probe_map_command)
                reward = self._normalized_reward(prev_snapshot.cost - final_snapshot.cost, terminal=True)
                sequence_str = self._sequence_str(tuple(next_sequence), self.probe_map_command)
                return {
                    "next_sequence": tuple(next_sequence),
                    "next_snapshot": final_snapshot,
                    "reward": reward,
                    "done": True,
                    "done_reason": "max_steps",
                    "sequence_str": sequence_str,
                }

            next_snapshot = self._evaluate(tuple(next_sequence), self.probe_map_command)
            reward = self._normalized_reward(prev_snapshot.cost - next_snapshot.cost, terminal=False)
            sequence_str = self._sequence_str(tuple(next_sequence), None)
            return {
                "next_sequence": tuple(next_sequence),
                "next_snapshot": next_snapshot,
                "reward": reward,
                "done": False,
                "done_reason": None,
                "sequence_str": sequence_str,
            }
        except subprocess.TimeoutExpired:
            return self._failed_transition(
                state=state,
                action=action,
                reason="action_timeout",
            )
        except RuntimeError:
            return self._failed_transition(
                state=state,
                action=action,
                reason="action_error",
            )

    def _evaluate(self, sequence: tuple[str, ...], map_command: str) -> EvalSnapshot:
        if not self.input_aig.is_file():
            raise FileNotFoundError(self.input_aig)
        if not self.imap_bin.is_file():
            raise FileNotFoundError(self.imap_bin)

        needs_history = self._map_uses_history(map_command)
        cache_key = (str(self.input_aig), sequence, map_command, needs_history)
        cached = self._EVAL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        commands = [f"read_aiger -f {self.input_aig}"]
        commands.extend(self._expanded_sequence(sequence, use_history=needs_history))
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

        aig_stats = self._parse_stats_line(proc.stdout, "Stats of AIG:")
        fpga_stats = self._parse_stats_line(proc.stdout, "Stats of FPGA:")
        if aig_stats is None or fpga_stats is None:
            raise RuntimeError(f"Failed to parse stats from iMAP output:\n{proc.stdout}")
        snapshot = EvalSnapshot(aig=aig_stats, fpga=fpga_stats, sequence=sequence)
        if len(self._EVAL_CACHE) >= self._CACHE_MAX_ENTRIES:
            self._EVAL_CACHE.clear()
        self._EVAL_CACHE[cache_key] = snapshot
        return snapshot

    def _map_uses_history(self, map_command: str | None) -> bool:
        return bool(self.use_history and map_command is not None and "-t 1" in map_command)

    def _expanded_sequence(
        self,
        sequence: tuple[str, ...],
        final_map_command: str | None = None,
        use_history: bool | None = None,
    ) -> list[str]:
        commands: list[str] = []
        if use_history is None:
            use_history = self._map_uses_history(final_map_command)
        if not use_history:
            commands.extend(sequence)
            if final_map_command is not None:
                commands.append(final_map_command)
            return commands

        commands.extend(["history -c", "history -a"])
        history_size = 1
        for command in sequence:
            commands.append(command)
            if history_size < self.history_capacity:
                commands.append("history -a")
                history_size += 1
        if final_map_command is not None:
            commands.append(final_map_command)
        return commands

    def _sequence_str(self, sequence: tuple[str, ...], final_map_command: str | None) -> str:
        commands = self._expanded_sequence(sequence, final_map_command)
        return "; ".join(commands) + ";"

    def _parse_stats_line(self, stdout: str, prefix: str) -> NetStats | None:
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            payload = stripped[len(prefix) :].strip()
            fields: dict[str, int] = {}
            for part in payload.split(","):
                item = part.strip()
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                try:
                    fields[key.strip()] = int(value.strip())
                except ValueError:
                    continue
            if {"pis", "pos", "area", "depth"} - fields.keys():
                return None
            return NetStats(
                pis=fields["pis"],
                pos=fields["pos"],
                area=fields["area"],
                depth=fields["depth"],
                inv=fields.get("inv", 0),
                po_inv=fields.get("po_inv", 0),
                fanout_sum=fields.get("fanout_sum", 0),
                fanout_max=fields.get("fanout_max", 0),
                fanout_ge4=fields.get("fanout_ge4", 0),
                level_sum=fields.get("level_sum", 0),
                level_ge_half=fields.get("level_ge_half", 0),
                lut1=fields.get("lut1", 0),
                lut2=fields.get("lut2", 0),
                lut3=fields.get("lut3", 0),
                lut4=fields.get("lut4", 0),
                lut5=fields.get("lut5", 0),
                lut6=fields.get("lut6", 0),
            )
        return None

    def _encode_state(
        self,
        state: SearchState,
    ) -> np.ndarray:
        if self.initial_snapshot is None:
            raise RuntimeError("Environment has not been reset.")

        snapshot = state.snapshot
        init = self.initial_snapshot
        step_denom = max(1.0, float(self.max_steps))
        action_count_scale = max(1.0, float(state.step_index))
        last_action_one_hot = np.zeros(len(self.actions) + 1, dtype=np.float32)
        last_action_slot = state.last_action_index if state.last_action_index >= 0 else len(self.actions)
        last_action_one_hot[last_action_slot] = 1.0

        def ratio(curr: float, base: float) -> float:
            return float(curr) / max(1.0, float(base))

        def delta(curr: float, base: float) -> float:
            return (float(curr) - float(base)) / max(1.0, float(base))

        def stats_features(stats: NetStats) -> list[float]:
            lut_hist = stats.lut_hist
            return [
                math.log1p(float(stats.pis)),
                math.log1p(float(stats.pos)),
                math.log1p(float(stats.area)),
                math.log1p(float(stats.depth)),
                math.log1p(float(stats.fanout_max)),
                stats.fanout_avg,
                stats.high_fanout_ratio,
                stats.level_avg,
                stats.high_level_ratio,
                stats.inv_ratio,
                stats.po_inv_ratio,
                *(float(count) / max(1.0, float(stats.area)) for count in lut_hist),
            ]

        current_features = np.array(
            [
                math.log1p(float(snapshot.cost)),
                *stats_features(snapshot.aig),
                *stats_features(snapshot.fpga),
            ],
            dtype=np.float32,
        )
        init_features = np.array(
            [
                math.log1p(float(init.cost)),
                *stats_features(init.aig),
                *stats_features(init.fpga),
            ],
            dtype=np.float32,
        )
        relational = np.array(
            [
                ratio(snapshot.aig.area, init.aig.area),
                ratio(snapshot.aig.depth, init.aig.depth),
                ratio(snapshot.fpga.area, init.fpga.area),
                ratio(snapshot.fpga.depth, init.fpga.depth),
                ratio(snapshot.cost, init.cost),
                delta(snapshot.aig.area, init.aig.area),
                delta(snapshot.aig.depth, init.aig.depth),
                delta(snapshot.fpga.area, init.fpga.area),
                delta(snapshot.fpga.depth, init.fpga.depth),
                delta(snapshot.cost, init.cost),
                snapshot.aig.inv_ratio - init.aig.inv_ratio,
                snapshot.aig.high_fanout_ratio - init.aig.high_fanout_ratio,
                snapshot.aig.high_level_ratio - init.aig.high_level_ratio,
                snapshot.fpga.high_fanout_ratio - init.fpga.high_fanout_ratio,
                snapshot.fpga.high_level_ratio - init.fpga.high_level_ratio,
                snapshot.fpga.level_avg - init.fpga.level_avg,
                snapshot.fpga.fanout_avg - init.fpga.fanout_avg,
                float(init.cost - snapshot.cost) / max(1.0, float(init.cost)),
                float(init.fpga.area - snapshot.fpga.area) / max(1.0, float(init.fpga.area)),
                float(init.fpga.depth - snapshot.fpga.depth) / max(1.0, float(init.fpga.depth)),
                float(state.step_index) / step_denom,
                float(self.max_steps - state.step_index) / step_denom,
            ],
            dtype=np.float32,
        )
        normalized_action_counts = state.action_counts.astype(np.float32) / action_count_scale
        return np.concatenate(
            [current_features, init_features, relational, normalized_action_counts, last_action_one_hot],
            axis=0,
        )
