#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import (
    IMAP_BIN,
    PrefixState,
    feature_vector,
    get_aig_stats,
    read_jsonl,
    split_by_hash,
    tensorize_features,
    evaluate_prefix_best_map,
)

def aggregate_targets(records: list[dict]) -> dict[tuple[str, tuple[str, ...], str], float]:
    best_after_action: dict[tuple[str, tuple[str, ...], str], float] = {}
    for record in records:
        case_path = record["case_path"]
        steps = tuple(record["step_labels"])
        final_cost = float(record["cost"])
        for idx, action_name in enumerate(steps):
            prefix = steps[:idx]
            key = (case_path, prefix, action_name)
            old = best_after_action.get(key)
            if old is None or final_cost < old:
                best_after_action[key] = final_cost
    return best_after_action


def build_prefix_cache(
    records: list[dict],
    best_after_action: dict[tuple[str, tuple[str, ...], str], float],
    *,
    imap_bin: Path,
    use_history: bool,
    timeout: float | None,
) -> tuple[dict[str, tuple[int, int]], dict[tuple[str, tuple[str, ...]], PrefixState]]:
    original_stats: dict[str, tuple[int, int]] = {}
    prefix_cache: dict[tuple[str, tuple[str, ...]], PrefixState] = {}

    needed_prefixes = {(case_path, prefix) for case_path, prefix, _ in best_after_action}
    print(f"evaluating unique prefixes: {len(needed_prefixes)}", flush=True)
    for index, (case_path, prefix) in enumerate(sorted(needed_prefixes), start=1):
        aig_path = Path(case_path)
        if case_path not in original_stats:
            original_stats[case_path] = get_aig_stats(aig_path, imap_bin=imap_bin)
        state = evaluate_prefix_best_map(
            aig_path=aig_path,
            prefix=prefix,
            use_history=use_history,
            imap_bin=imap_bin,
            timeout=timeout,
        )
        if state is not None:
            prefix_cache[(case_path, prefix)] = state
        if index % 200 == 0:
            print(f"  prefix progress {index}/{len(needed_prefixes)}", flush=True)
    return original_stats, prefix_cache


def build_examples(
    records: list[dict],
    best_after_action: dict[tuple[str, tuple[str, ...], str], float],
    original_stats: dict[str, tuple[int, int]],
    prefix_cache: dict[tuple[str, tuple[str, ...]], PrefixState],
) -> list[dict]:
    examples = []
    seen = set()
    for case_path, prefix, action_name in sorted(best_after_action):
        state = prefix_cache.get((case_path, prefix))
        if state is None:
            continue
        original_area, original_depth = original_stats[case_path]
        best_future_cost = best_after_action[(case_path, prefix, action_name)]
        target_improvement = (state.cost - best_future_cost) / max(state.cost, 1.0)
        case_name = Path(case_path).stem
        key = (case_path, prefix, action_name)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "case_name": case_name,
                "case_path": case_path,
                "prefix": list(prefix),
                "action_name": action_name,
                "state_area": state.area,
                "state_depth": state.depth,
                "state_cost": state.cost,
                "best_future_cost": best_future_cost,
                "target_improvement": target_improvement,
                "features": feature_vector(
                    original_area=original_area,
                    original_depth=original_depth,
                    state=state,
                    action_name=action_name,
                ),
            }
        )
    return examples


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> dict:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_xn = (train_x - mean) / std
    val_xn = (val_x - mean) / std

    train_aug = np.concatenate([train_xn, np.ones((train_xn.shape[0], 1))], axis=1)
    val_aug = np.concatenate([val_xn, np.ones((val_xn.shape[0], 1))], axis=1)

    best = None
    history = []
    for reg in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        reg_eye = np.eye(train_aug.shape[1], dtype=np.float64) * reg
        reg_eye[-1, -1] = 0.0
        weights = np.linalg.solve(train_aug.T @ train_aug + reg_eye, train_aug.T @ train_y)
        train_pred = train_aug @ weights
        val_pred = val_aug @ weights
        train_rmse = float(np.sqrt(np.mean((train_pred - train_y) ** 2)))
        val_rmse = float(np.sqrt(np.mean((val_pred - val_y) ** 2)))
        history.append({"reg": reg, "train_rmse": train_rmse, "val_rmse": val_rmse})
        print(
            f"reg={reg:.4g} train_rmse={train_rmse:.5f} val_rmse={val_rmse:.5f}",
            flush=True,
        )
        if best is None or val_rmse < best["val_rmse"]:
            best = {
                "weights": weights[:-1],
                "bias": float(weights[-1]),
                "mean": mean,
                "std": std,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "reg": reg,
            }
    assert best is not None
    best["history"] = history
    best["feature_dim"] = int(train_x.shape[1])
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a value model from rollout data.")
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--imap-bin", type=Path, default=IMAP_BIN)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    started = time.time()
    records = read_jsonl(args.rollout_jsonl.resolve())
    if not records:
        raise SystemExit("rollout dataset is empty")
    use_history = bool(records[0].get("use_history", True))
    best_after_action = aggregate_targets(records)
    original_stats, prefix_cache = build_prefix_cache(
        records,
        best_after_action,
        imap_bin=args.imap_bin.resolve(),
        use_history=use_history,
        timeout=args.timeout,
    )
    examples = build_examples(records, best_after_action, original_stats, prefix_cache)
    if len(examples) < 20:
        raise SystemExit(f"not enough examples: {len(examples)}")

    case_names = [example["case_name"] for example in examples]
    val_flags = split_by_hash(case_names)
    train_rows = [example for example, is_val in zip(examples, val_flags) if not is_val]
    val_rows = [example for example, is_val in zip(examples, val_flags) if is_val]
    if not val_rows:
        val_rows = train_rows[-max(1, len(train_rows) // 10) :]
        train_rows = train_rows[: -len(val_rows)]

    train_x = tensorize_features([row["features"] for row in train_rows])
    train_y = np.asarray([row["target_improvement"] for row in train_rows], dtype=np.float64)
    val_x = tensorize_features([row["features"] for row in val_rows])
    val_y = np.asarray([row["target_improvement"] for row in val_rows], dtype=np.float64)
    best = train_model(
        train_x,
        train_y,
        val_x,
        val_y,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_dim": best["feature_dim"],
        "best_val_rmse": best["val_rmse"],
        "best_train_rmse": best["train_rmse"],
        "best_reg": best["reg"],
        "history": best["history"],
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "use_history": use_history,
        "source_rollouts": str(args.rollout_jsonl.resolve()),
        "elapsed_sec": time.time() - started,
    }
    np.savez(
        args.output_model.resolve(),
        weights=best["weights"],
        bias=np.asarray(best["bias"], dtype=np.float64),
        mean=best["mean"],
        std=best["std"],
    )
    args.output_model.with_suffix(args.output_model.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.output_model.with_suffix(args.output_model.suffix + ".examples.json").write_text(
        json.dumps(
            {
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "elapsed_sec": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
