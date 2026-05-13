from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import FlowAction


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


@dataclass
class StateStats:
    aig: NetStats
    sequence: tuple[str, ...]
    step_index: int
    last_action_index: int
    action_counts: np.ndarray


@dataclass
class FinalStats:
    area: int
    depth: int

    @property
    def cost(self) -> float:
        return 0.6 * self.depth + 0.4 * self.area


class AIGEnv:
    _AIG_CACHE: dict[tuple[str, tuple[str, ...]], NetStats] = {}
    _FINAL_CACHE: dict[tuple[str, tuple[str, ...], str], FinalStats] = {}

    def __init__(
        self,
        input_aig: Path,
        imap_bin: Path,
        actions: list[FlowAction],
        max_steps: int = 4,
        timeout_sec: float = 60.0,
        final_timeout_sec: float | None = None,
        use_history: bool = True,
        history_capacity: int = 5,
    ) -> None:
        self.input_aig = Path(input_aig).resolve()
        self.imap_bin = Path(imap_bin).resolve()
        self.actions = actions
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        self.final_timeout_sec = final_timeout_sec if final_timeout_sec is not None else timeout_sec
        self.use_history = use_history
        self.history_capacity = history_capacity
        self.initial_state: StateStats | None = None

    def reset(self) -> np.ndarray:
        aig = self.evaluate_aig(tuple())
        self.initial_state = StateStats(
            aig=aig,
            sequence=tuple(),
            step_index=0,
            last_action_index=-1,
            action_counts=np.zeros(len(self.actions), dtype=np.float32),
        )
        return self.observe(self.initial_state)

    def observe(self, state: StateStats) -> np.ndarray:
        if self.initial_state is None:
            raise RuntimeError("reset must be called before observe")
        init = self.initial_state.aig
        curr = state.aig
        last_one_hot = np.zeros(len(self.actions) + 1, dtype=np.float32)
        last_slot = state.last_action_index if state.last_action_index >= 0 else len(self.actions)
        last_one_hot[last_slot] = 1.0
        step_denom = max(1.0, float(self.max_steps))
        counts = state.action_counts.astype(np.float32) / max(1.0, float(state.step_index))

        def stats_features(stats: NetStats) -> list[float]:
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
            ]

        def ratio(curr_value: float, init_value: float) -> float:
            return float(curr_value) / max(1.0, float(init_value))

        def delta(curr_value: float, init_value: float) -> float:
            return (float(curr_value) - float(init_value)) / max(1.0, float(init_value))

        obs = np.array(
            [
                *stats_features(curr),
                *stats_features(init),
                ratio(curr.area, init.area),
                ratio(curr.depth, init.depth),
                delta(curr.area, init.area),
                delta(curr.depth, init.depth),
                curr.fanout_avg - init.fanout_avg,
                curr.level_avg - init.level_avg,
                curr.high_fanout_ratio - init.high_fanout_ratio,
                curr.high_level_ratio - init.high_level_ratio,
                curr.inv_ratio - init.inv_ratio,
                curr.po_inv_ratio - init.po_inv_ratio,
                float(state.step_index) / step_denom,
                float(self.max_steps - state.step_index) / step_denom,
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs, counts, last_one_hot], axis=0)

    def next_state(self, state: StateStats, action_index: int) -> StateStats:
        action = self.actions[action_index]
        next_sequence = tuple(list(state.sequence) + list(action.commands))
        next_counts = state.action_counts.copy()
        next_counts[action_index] += 1.0
        aig = state.aig if action.terminal else self.evaluate_aig(next_sequence)
        return StateStats(
            aig=aig,
            sequence=state.sequence if action.terminal else next_sequence,
            step_index=state.step_index + 1,
            last_action_index=action_index,
            action_counts=next_counts,
        )

    def evaluate_aig(self, sequence: tuple[str, ...]) -> NetStats:
        cache_key = (str(self.input_aig), sequence)
        cached = self._AIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        commands = [f"read_aiger -f {self.input_aig}"]
        commands.extend(sequence)
        commands.append("print_stats -t 0")
        script = "; ".join(commands) + ";"
        proc = subprocess.run(
            [str(self.imap_bin), "-c", script],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        stats = self._parse_aig_stats(proc.stdout)
        self._AIG_CACHE[cache_key] = stats
        return stats

    def evaluate_final(self, sequence: tuple[str, ...], map_command: str) -> FinalStats:
        cache_key = (str(self.input_aig), sequence, map_command)
        cached = self._FINAL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        commands = [f"read_aiger -f {self.input_aig}"]
        commands.extend(self._expanded_sequence(sequence, map_command))
        commands.append("print_stats -t 1")
        script = "; ".join(commands) + ";"
        proc = subprocess.run(
            [str(self.imap_bin), "-c", script],
            capture_output=True,
            text=True,
            timeout=self.final_timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        stats = self._parse_final_stats(proc.stdout)
        self._FINAL_CACHE[cache_key] = stats
        return stats

    def _expanded_sequence(self, sequence: tuple[str, ...], final_map_command: str) -> list[str]:
        if not self.use_history or "-t 1" not in final_map_command:
            return [*sequence, final_map_command]
        commands = ["history -c", "history -a"]
        history_size = 1
        for command in sequence:
            commands.append(command)
            if history_size < self.history_capacity:
                commands.append("history -a")
                history_size += 1
        commands.append(final_map_command)
        return commands

    def _parse_aig_stats(self, stdout: str) -> NetStats:
        prefix = "Stats of AIG:"
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith(prefix):
                continue
            fields = self._parse_key_values(line[len(prefix):])
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
            )
        raise RuntimeError(f"failed to parse AIG stats from output:\n{stdout}")

    def _parse_final_stats(self, stdout: str) -> FinalStats:
        prefix = "Stats of FPGA:"
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith(prefix):
                continue
            fields = self._parse_key_values(line[len(prefix):])
            return FinalStats(area=fields["area"], depth=fields["depth"])
        raise RuntimeError(f"failed to parse FPGA stats from output:\n{stdout}")

    @staticmethod
    def _parse_key_values(payload: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for part in payload.split(","):
            item = part.strip()
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            result[key.strip()] = int(value.strip())
        return result
