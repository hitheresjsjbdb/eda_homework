from __future__ import annotations

import math


def case_weight(init_cost: float, min_weight: float = 1.0, max_weight: float = 1.8) -> float:
    scaled = math.log1p(max(1.0, init_cost)) / math.log1p(1000.0)
    weight = min_weight + (max_weight - min_weight) * max(0.0, min(1.0, scaled))
    return weight


def case_bucket(
    init_cost: float,
    small_threshold: float = 200.0,
    large_threshold: float = 1000.0,
) -> str:
    if init_cost < small_threshold:
        return "small"
    if init_cost < large_threshold:
        return "medium"
    return "large"


def case_bucket_index(
    init_cost: float,
    small_threshold: float = 200.0,
    large_threshold: float = 1000.0,
) -> int:
    bucket = case_bucket(init_cost, small_threshold=small_threshold, large_threshold=large_threshold)
    if bucket == "small":
        return 0
    if bucket == "medium":
        return 1
    return 2


def adapt_search_budget(
    init_cost: float,
    base_max_steps: int,
    base_beam_width: int,
    base_branch_topk: int,
) -> tuple[int, int, int]:
    if init_cost >= 1000:
        return min(base_max_steps + 2, 6), max(2, base_beam_width - 2), max(2, base_branch_topk - 1)
    if init_cost >= 500:
        return min(base_max_steps + 1, 6), max(2, base_beam_width - 1), max(2, base_branch_topk - 1)
    return base_max_steps, base_beam_width, base_branch_topk
