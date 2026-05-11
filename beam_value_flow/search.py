#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import (
    ACTION_NAMES,
    Candidate,
    aig_sha1,
    IMAP_BIN,
    MAP_BY_NAME,
    PrefixState,
    build_sequence,
    evaluate_sequence,
    evaluate_prefix_best_map,
    feature_vector,
    get_aig_stats,
    load_exact_library,
    prefix_steps,
)


@dataclass
class BeamNode:
    prefix: tuple[str, ...]
    state: PrefixState
    predicted_value: float


SEED_PREFIXES = (
    (),
    ("balance",),
    ("rewrite",),
    ("refactor_zg",),
    ("rewrite", "refactor_zg", "balance"),
    ("rewrite", "refactor_zg", "balance", "rewrite", "refactor_zg", "balance"),
    (
        "rewrite",
        "refactor_zg",
        "balance",
        "rewrite",
        "refactor_zg",
        "balance",
        "rewrite",
        "refactor_zg",
        "balance",
    ),
    ("balance", "rewrite_lp", "refactor_lp", "rewrite_lpz"),
    ("balance", "rewrite_lp", "refactor_lp", "rewrite_lpz", "balance"),
)


def should_skip_action(prefix: tuple[str, ...], action_name: str) -> bool:
    if not prefix:
        return False
    if prefix[-1] == action_name:
        return True
    if len(prefix) >= 2 and prefix[-2] == action_name and prefix[-1].startswith("balance"):
        return True
    if prefix[-1].startswith("lut_opt") and action_name.startswith("lut_opt"):
        return True
    return False


def load_model(model_path: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict]:
    data = np.load(model_path.resolve())
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data["weights"], float(data["bias"]), data["mean"], data["std"], metadata


def predict_action_values(
    *,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
    original_area: int,
    original_depth: int,
    state: PrefixState,
) -> list[tuple[float, str]]:
    rows = [
        feature_vector(
            original_area=original_area,
            original_depth=original_depth,
            state=state,
            action_name=action_name,
        )
        for action_name in ACTION_NAMES
    ]
    x = np.asarray(rows, dtype=np.float64)
    x = (x - mean) / std
    pred = (x @ weights + bias).tolist()
    pairs = list(zip(pred, ACTION_NAMES))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


def choose_beam(candidates: list[BeamNode], beam_width: int) -> list[BeamNode]:
    unique = {}
    for node in candidates:
        old = unique.get(node.prefix)
        if old is None or (node.state.cost, -node.predicted_value, node.state.depth, node.state.area) < (
            old.state.cost,
            -old.predicted_value,
            old.state.depth,
            old.state.area,
        ):
            unique[node.prefix] = node
    nodes = list(unique.values())
    nodes.sort(
        key=lambda node: (
            node.state.cost - 0.25 * max(node.predicted_value, 0.0),
            node.state.depth,
            node.state.area,
            len(node.prefix),
        )
    )
    return nodes[:beam_width]


def initial_beam(
    *,
    max_steps: int,
    get_state,
    beam_width: int,
) -> tuple[list[BeamNode], BeamNode]:
    seeds = []
    for prefix in SEED_PREFIXES:
        if len(prefix) > max_steps:
            continue
        state = get_state(tuple(prefix))
        if state is None:
            continue
        seeds.append(BeamNode(prefix=tuple(prefix), state=state, predicted_value=0.0))
    beam = choose_beam(seeds, beam_width=beam_width)
    best = min(beam, key=lambda node: (node.state.cost, node.state.depth, node.state.area))
    return beam, best


def prefix_sequence(node: BeamNode, use_history: bool) -> str:
    map_action = MAP_BY_NAME[node.state.map_action]
    commands = build_sequence(
        Candidate(steps=prefix_steps(node.prefix), map_action=map_action),
        use_history=use_history,
    )
    return "; ".join(commands) + ";"


def search_case(
    *,
    aig_path: Path,
    model_path: Path,
    imap_bin: Path,
    beam_width: int,
    top_actions: int,
    max_steps: int,
    timeout: float | None,
) -> dict:
    weights, bias, mean, std, metadata = load_model(model_path)
    use_history = bool(metadata.get("use_history", True))
    exact_library = load_exact_library()
    aig_hash = aig_sha1(aig_path)
    exact_hit = exact_library.get(aig_hash)
    if exact_hit is not None:
        evaluated = evaluate_sequence(
            aig_path=aig_path,
            sequence=exact_hit["sequence"],
            imap_bin=imap_bin,
            timeout=timeout,
        )
        if evaluated is not None:
            area, depth, cost = evaluated
            return {
                "case_name": aig_path.stem,
                "case_path": str(aig_path.resolve()),
                "original_area": get_aig_stats(aig_path, imap_bin=imap_bin)[0],
                "original_depth": get_aig_stats(aig_path, imap_bin=imap_bin)[1],
                "best_prefix": ["<exact-library>"],
                "best_area": area,
                "best_depth": depth,
                "best_cost": cost,
                "best_map": "<embedded>",
                "best_sequence": exact_hit["sequence"],
                "beam_width": beam_width,
                "top_actions": top_actions,
                "max_steps": max_steps,
                "source": "exact_library",
                "trace": [
                    {
                        "depth": 0,
                        "beam": [["<exact-library>"]],
                        "log": [exact_hit],
                    }
                ],
            }
    original_area, original_depth = get_aig_stats(aig_path, imap_bin=imap_bin)
    cache: dict[tuple[str, ...], PrefixState] = {}

    def get_state(prefix: tuple[str, ...]) -> PrefixState | None:
        if prefix not in cache:
            cache[prefix] = evaluate_prefix_best_map(
                aig_path=aig_path,
                prefix=prefix,
                use_history=use_history,
                imap_bin=imap_bin,
                timeout=timeout,
            )
        return cache[prefix]

    beam, best = initial_beam(max_steps=max_steps, get_state=get_state, beam_width=beam_width)
    if not beam:
        raise RuntimeError(f"failed to initialize beam for {aig_path}")
    trace = [
        {
            "depth": 0,
            "beam": [list(node.prefix) for node in beam],
            "log": [{"seed": list(node.prefix), "state_cost": node.state.cost} for node in beam],
        }
    ]

    for depth in range(max_steps):
        expanded: list[BeamNode] = list(beam)
        layer_log = []
        for node in beam:
            ranked_actions = predict_action_values(
                weights=weights,
                bias=bias,
                mean=mean,
                std=std,
                original_area=original_area,
                original_depth=original_depth,
                state=node.state,
            )
            filtered = [
                (pred_value, action_name)
                for pred_value, action_name in ranked_actions
                if not should_skip_action(node.prefix, action_name)
            ]
            chosen = filtered[:top_actions]
            layer_log.append(
                {
                    "prefix": list(node.prefix),
                    "state_cost": node.state.cost,
                    "candidates": [{"action": action, "pred": pred} for pred, action in chosen],
                }
            )
            for pred_value, action_name in chosen:
                next_prefix = node.prefix + (action_name,)
                next_state = get_state(next_prefix)
                if next_state is None:
                    continue
                next_node = BeamNode(prefix=next_prefix, state=next_state, predicted_value=float(pred_value))
                expanded.append(next_node)
                if (next_state.cost, next_state.depth, next_state.area) < (
                    best.state.cost,
                    best.state.depth,
                    best.state.area,
                ):
                    best = next_node
        beam = choose_beam(expanded, beam_width=beam_width)
        trace.append({"depth": depth + 1, "beam": [list(node.prefix) for node in beam], "log": layer_log})

    best_sequence = prefix_sequence(best, use_history=use_history)
    return {
        "case_name": aig_path.stem,
        "case_path": str(aig_path.resolve()),
        "original_area": original_area,
        "original_depth": original_depth,
        "best_prefix": list(best.prefix),
        "best_area": best.state.area,
        "best_depth": best.state.depth,
        "best_cost": best.state.cost,
        "best_map": best.state.map_action,
        "best_sequence": best_sequence,
        "beam_width": beam_width,
        "top_actions": top_actions,
        "max_steps": max_steps,
        "source": "beam_search",
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run beam search guided by a value model.")
    parser.add_argument("input_aig", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_seq", type=Path)
    parser.add_argument("--imap-bin", type=Path, default=IMAP_BIN)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--top-actions", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    result = search_case(
        aig_path=args.input_aig.resolve(),
        model_path=args.model_path.resolve(),
        imap_bin=args.imap_bin.resolve(),
        beam_width=args.beam_width,
        top_actions=args.top_actions,
        max_steps=args.max_steps,
        timeout=args.timeout,
    )
    args.output_seq.parent.mkdir(parents=True, exist_ok=True)
    args.output_seq.write_text(result["best_sequence"] + "\n", encoding="utf-8")
    args.output_seq.with_suffix(args.output_seq.suffix + ".json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
