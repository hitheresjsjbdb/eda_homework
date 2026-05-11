from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAP_ROOT = REPO_ROOT / "iMAP"
IMAP_BIN = IMAP_ROOT / "bin" / "imap"
EDA23_ROOT = IMAP_ROOT / "eda23"
EXACT_LIBRARY_PATH = REPO_ROOT / "beam_value_flow" / "exact_library.json"

import sys

if str(EDA23_ROOT) not in sys.path:
    sys.path.insert(0, str(EDA23_ROOT))

from bruteforce_search import (  # noqa: E402
    DEFAULT_ACTIONS,
    DEFAULT_MAP_ACTIONS,
    Action,
    Candidate,
    EvalResult,
    build_sequence,
    evaluate_candidate,
)


AIG_STATS_RE = re.compile(r"Stats of AIG:.*?area=(\d+), depth=(\d+)")

ACTION_NAMES = [action.name for action in DEFAULT_ACTIONS]
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
MAP_NAMES = [action.name for action in DEFAULT_MAP_ACTIONS]
MAP_INDEX = {name: idx for idx, name in enumerate(MAP_NAMES)}
ACTION_BY_NAME = {action.name: action for action in DEFAULT_ACTIONS}
MAP_BY_NAME = {action.name: action for action in DEFAULT_MAP_ACTIONS}


@dataclass(frozen=True)
class PrefixState:
    case_name: str
    case_path: str
    prefix: tuple[str, ...]
    area: int
    depth: int
    cost: float
    map_action: str


def discover_aigs(root: Path) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        if root.suffix != ".aig":
            raise SystemExit(f"expected .aig file, got: {root}")
        return [root]
    if not root.is_dir():
        raise SystemExit(f"path not found: {root}")
    return sorted(root.rglob("*.aig"))


def aig_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def get_aig_stats(aig_path: Path, imap_bin: Path = IMAP_BIN) -> tuple[int, int]:
    import subprocess

    proc = subprocess.run(
        [str(imap_bin), "-c", f"read_aiger -f {aig_path}; print_stats -t 0;"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = AIG_STATS_RE.search(proc.stdout or "")
    if proc.returncode != 0 or match is None:
        raise RuntimeError(f"failed to read stats for {aig_path}: {proc.stderr}")
    return int(match.group(1)), int(match.group(2))


def evaluate_sequence(
    aig_path: Path,
    sequence: str,
    *,
    imap_bin: Path = IMAP_BIN,
    timeout: float | None = None,
) -> tuple[int, int, float] | None:
    import subprocess

    flow = sequence.strip()
    if not flow.endswith(";"):
        flow += ";"
    if "read_aiger" not in flow:
        flow = f"read_aiger -f {aig_path}; " + flow
    flow += " print_stats -t 1;"
    try:
        proc = subprocess.run(
            [str(imap_bin), "-c", flow],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"Stats of FPGA:.*?area=(\d+), depth=(\d+)", proc.stdout or "")
    if match is None:
        return None
    area = int(match.group(1))
    depth = int(match.group(2))
    return area, depth, 0.4 * area + 0.6 * depth


def load_exact_library(path: Path = EXACT_LIBRARY_PATH) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_steps(prefix: tuple[str, ...]) -> tuple[Action, ...]:
    return tuple(ACTION_BY_NAME[name] for name in prefix)


def evaluate_prefix_best_map(
    aig_path: Path,
    prefix: tuple[str, ...],
    use_history: bool,
    imap_bin: Path = IMAP_BIN,
    timeout: float | None = None,
) -> PrefixState | None:
    import subprocess

    best = None
    steps = prefix_steps(prefix)
    for map_action in DEFAULT_MAP_ACTIONS:
        try:
            result = evaluate_candidate(
                aig_path=aig_path,
                imap_bin=imap_bin,
                candidate=Candidate(steps=steps, map_action=map_action),
                use_history=use_history,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result = None
        if result is None:
            continue
        if best is None or (result.cost, result.depth, result.area) < (
            best.cost,
            best.depth,
            best.area,
        ):
            best = result
    if best is None:
        return None
    return PrefixState(
        case_name=aig_path.stem,
        case_path=str(aig_path),
        prefix=prefix,
        area=best.area,
        depth=best.depth,
        cost=best.cost,
        map_action=best.candidate.map_action.name,
    )


def prefix_meta_features(prefix: tuple[str, ...]) -> list[float]:
    length = len(prefix)
    counts = [0.0] * len(ACTION_NAMES)
    for name in prefix:
        counts[ACTION_INDEX[name]] += 1.0
    if length > 0:
        counts = [value / length for value in counts]

    def last_action_features(offset: int) -> list[float]:
        feats = [0.0] * (len(ACTION_NAMES) + 1)
        if length > offset:
            feats[ACTION_INDEX[prefix[-1 - offset]]] = 1.0
        else:
            feats[-1] = 1.0
        return feats

    return [float(length)] + counts + last_action_features(0) + last_action_features(1)


def feature_vector(
    *,
    original_area: int,
    original_depth: int,
    state: PrefixState,
    action_name: str,
) -> list[float]:
    prefix = state.prefix
    feats = [
        math.log1p(original_area),
        math.log1p(original_depth),
        math.log1p(state.area),
        math.log1p(state.depth),
        state.cost,
        state.area / max(original_area, 1),
        state.depth / max(original_depth, 1),
        (state.cost - (0.4 * original_area + 0.6 * original_depth))
        / max(0.4 * original_area + 0.6 * original_depth, 1.0),
    ]
    feats.extend(prefix_meta_features(prefix))
    action_one_hot = [0.0] * len(ACTION_NAMES)
    action_one_hot[ACTION_INDEX[action_name]] = 1.0
    feats.extend(action_one_hot)
    return feats


def tensorize_features(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def split_by_hash(case_names: list[str], mod: int = 10) -> list[bool]:
    flags = []
    for case_name in case_names:
        value = int(hashlib.sha1(case_name.encode("utf-8")).hexdigest(), 16)
        flags.append((value % mod) == 0)
    return flags
