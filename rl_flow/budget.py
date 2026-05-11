from __future__ import annotations

import math


def case_weight(init_cost: float, min_weight: float = 1.0, max_weight: float = 3.0) -> float:
    scaled = math.log1p(max(1.0, init_cost)) / math.log1p(1000.0)
    weight = min_weight + (max_weight - min_weight) * max(0.0, min(1.0, scaled))
    return weight


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
