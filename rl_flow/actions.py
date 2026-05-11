from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroAction:
    name: str
    commands: tuple[str, ...]
    terminal: bool = False
    final_map_command: str | None = None


def default_macro_actions() -> list[MacroAction]:
    return [
        MacroAction("balance", ("balance",)),
        MacroAction("rewrite_lp", ("rewrite -l",)),
        MacroAction("rewrite_lpz", ("rewrite -l -z",)),
        MacroAction("rewrite_p12_lp", ("rewrite -P 12 -C 4 -l",)),
        MacroAction("refactor_lp", ("refactor -l",)),
        MacroAction("refactor_i12_c20_lp", ("refactor -I 12 -C 20 -l",)),
        MacroAction("lut_opt", ("lut_opt",)),
        MacroAction("lut_opt_area", ("lut_opt -P 12 -C 6 -G 2 -L 3",)),
        MacroAction("rw_bal", ("rewrite -l", "balance")),
        MacroAction("bal_rw", ("balance", "rewrite -l -z")),
        MacroAction("rw_ref", ("rewrite -l", "refactor -I 12 -C 20 -l")),
        MacroAction("lut_bal", ("lut_opt", "balance")),
        MacroAction(
            "stop_map_default",
            tuple(),
            terminal=True,
            final_map_command="map_fpga",
        ),
        MacroAction(
            "stop_map_cut12",
            tuple(),
            terminal=True,
            final_map_command="map_fpga -P 12 -C 6 -G 1 -L 2",
        ),
        MacroAction(
            "stop_map_area",
            tuple(),
            terminal=True,
            final_map_command="map_fpga -P 12 -C 6 -G 2 -L 3",
        ),
    ]
