from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowAction:
    name: str
    commands: tuple[str, ...]
    terminal: bool = False
    final_map_command: str | None = None
    teacher_priority: int = 0


def default_actions() -> list[FlowAction]:
    return [
        FlowAction("balance", ("balance",)),
        FlowAction("rewrite", ("rewrite",)),
        FlowAction("rewrite_lp", ("rewrite -l",)),
        FlowAction("rewrite_lpz", ("rewrite -l -z",)),
        FlowAction("rewrite_p12_lp", ("rewrite -P 12 -C 4 -l",)),
        FlowAction("rewrite_p12_lpz", ("rewrite -P 12 -C 4 -l -z",)),
        FlowAction("refactor", ("refactor",)),
        FlowAction("refactor_lp", ("refactor -l",)),
        FlowAction("refactor_lpz", ("refactor -l -z",)),
        FlowAction("refactor_i12_c20_lp", ("refactor -I 12 -C 20 -l",)),
        FlowAction("refactor_i8_c12_lp", ("refactor -I 8 -C 12 -l",)),
        FlowAction("lut_opt", ("lut_opt",)),
        FlowAction("lut_opt_zg", ("lut_opt -z",)),
        FlowAction("lut_opt_fast", ("lut_opt -P 12 -C 6 -G 1 -L 1",)),
        FlowAction("lut_opt_area", ("lut_opt -P 12 -C 6 -G 2 -L 3",)),
        FlowAction("lut_opt_area_zg", ("lut_opt -P 12 -C 6 -G 2 -L 3 -z",)),
        FlowAction("rw_bal", ("rewrite -l", "balance")),
        FlowAction("bal_rw", ("balance", "rewrite -l -z")),
        FlowAction("rw_ref", ("rewrite -l", "refactor -I 12 -C 20 -l")),
        FlowAction("ref_rw", ("refactor -l", "rewrite -l -z")),
        FlowAction("lut_bal", ("lut_opt", "balance")),
        FlowAction("bal_ref", ("balance", "refactor -l")),
        FlowAction("rwz_lut", ("rewrite -l -z", "lut_opt -P 12 -C 6 -G 2 -L 3")),
        FlowAction("rwz_bal_rw", ("rewrite -l -z", "balance", "rewrite -l")),
        FlowAction("rwz_bal_rwz", ("rewrite -l -z", "balance", "rewrite -l -z")),
        FlowAction("ref_bal_rwz", ("refactor -l", "balance", "rewrite -l -z")),
        FlowAction("rw_ref_bal", ("rewrite -l", "refactor -I 12 -C 20 -l", "balance")),
        FlowAction("p12rw_bal_ref", ("rewrite -P 12 -C 4 -l -z", "balance", "refactor -l")),
        FlowAction("ref_rwz_bal", ("refactor -l", "rewrite -l -z", "balance")),
        FlowAction("bal_ref_rwz", ("balance", "refactor -l", "rewrite -l -z")),
        FlowAction("lut_area_bal_rw", ("lut_opt -P 12 -C 6 -G 2 -L 3", "balance", "rewrite -l")),
        FlowAction("lut_fast_bal_rw", ("lut_opt -P 12 -C 6 -G 1 -L 1", "balance", "rewrite -l")),
        FlowAction("rwz_lut_bal", ("rewrite -l -z", "lut_opt -P 12 -C 6 -G 2 -L 3", "balance")),
        FlowAction("ref_rwz_lut", ("refactor -l", "rewrite -l -z", "lut_opt -P 12 -C 6 -G 2 -L 3")),
        FlowAction("p12rw_ref_lut", ("rewrite -P 12 -C 4 -l -z", "refactor -I 12 -C 20 -l", "lut_opt -P 12 -C 6 -G 2 -L 3")),
        FlowAction("rwz_bal_ref_rwz", ("rewrite -l -z", "balance", "refactor -l", "rewrite -l -z")),
        FlowAction("ref_bal_rwz_lut", ("refactor -l", "balance", "rewrite -l -z", "lut_opt -P 12 -C 6 -G 2 -L 3")),
        FlowAction("stop_map_fast", tuple(), terminal=True, final_map_command="map_fpga -P 10 -C 6 -G 1 -L 1", teacher_priority=3),
        FlowAction("stop_map_cut12", tuple(), terminal=True, final_map_command="map_fpga -P 12 -C 6 -G 1 -L 2", teacher_priority=4),
        FlowAction("stop_map_area", tuple(), terminal=True, final_map_command="map_fpga -P 12 -C 6 -G 2 -L 3", teacher_priority=2),
        FlowAction("stop_map_hist_cut12", tuple(), terminal=True, final_map_command="map_fpga -P 12 -C 6 -G 1 -L 2 -t 1", teacher_priority=5),
        FlowAction("stop_map_hist_area", tuple(), terminal=True, final_map_command="map_fpga -P 12 -C 6 -G 2 -L 3 -t 1", teacher_priority=1),
    ]
