#!/usr/bin/env python3
import argparse
import heapq
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from bruteforce_search import (
    Action,
    Candidate,
    EvalResult,
    build_sequence,
    candidate_to_labels,
    evaluate_candidate,
    serialize_sequence,
)


HEURISTIC_ACTIONS = (
    Action("balance", "balance"),
    Action("rewrite", "rewrite"),
    Action("rewrite_lp", "rewrite -l"),
    Action("rewrite_zg", "rewrite -z"),
    Action("rewrite_lpz", "rewrite -l -z"),
    Action("rewrite_c3", "rewrite -P 10 -C 3"),
    Action("rewrite_c3_lp", "rewrite -P 10 -C 3 -l"),
    Action("rewrite_p12", "rewrite -P 12 -C 4"),
    Action("rewrite_p12_lp", "rewrite -P 12 -C 4 -l"),
    Action("rewrite_p12_lpz", "rewrite -P 12 -C 4 -l -z"),
    Action("refactor", "refactor"),
    Action("refactor_lp", "refactor -l"),
    Action("refactor_zg", "refactor -z"),
    Action("refactor_lpz", "refactor -l -z"),
    Action("refactor_i12_c20", "refactor -I 12 -C 20"),
    Action("refactor_i12_c20_lp", "refactor -I 12 -C 20 -l"),
    Action("refactor_i8_c12", "refactor -I 8 -C 12"),
    Action("refactor_i8_c12_lp", "refactor -I 8 -C 12 -l"),
    Action("lut_opt", "lut_opt"),
    Action("lut_opt_zg", "lut_opt -z"),
    Action("lut_opt_c4", "lut_opt -P 10 -C 4 -G 1 -L 2"),
    Action("lut_opt_area", "lut_opt -P 12 -C 6 -G 2 -L 3"),
    Action("lut_opt_area_zg", "lut_opt -P 12 -C 6 -G 2 -L 3 -z"),
    Action("lut_opt_fast", "lut_opt -P 12 -C 6 -G 1 -L 1"),
)


FOCUSED_MUTATION_ACTIONS = (
    Action("balance", "balance"),
    Action("rewrite_lp", "rewrite -l"),
    Action("rewrite_lpz", "rewrite -l -z"),
    Action("rewrite_p12_lp", "rewrite -P 12 -C 4 -l"),
    Action("rewrite_p12_lpz", "rewrite -P 12 -C 4 -l -z"),
    Action("refactor_lp", "refactor -l"),
    Action("refactor_lpz", "refactor -l -z"),
    Action("refactor_i12_c20_lp", "refactor -I 12 -C 20 -l"),
    Action("lut_opt", "lut_opt"),
    Action("lut_opt_area", "lut_opt -P 12 -C 6 -G 2 -L 3"),
    Action("lut_opt_area_zg", "lut_opt -P 12 -C 6 -G 2 -L 3 -z"),
)


HEURISTIC_MAP_ACTIONS = (
    Action("map_default", "map_fpga"),
    Action("map_fast", "map_fpga -P 10 -C 6 -G 1 -L 1"),
    Action("map_cut12", "map_fpga -P 12 -C 6 -G 1 -L 2"),
    Action("map_area", "map_fpga -P 12 -C 6 -G 2 -L 3"),
    Action("map_area_mid", "map_fpga -P 10 -C 6 -G 2 -L 2"),
    Action("map_cut4", "map_fpga -P 12 -C 4 -G 1 -L 2"),
    Action("map_cut4_area", "map_fpga -P 12 -C 4 -G 2 -L 3"),
)


ACTION_MAP = {action.name: action for action in HEURISTIC_ACTIONS}


SEED_TEMPLATES = (
    (),
    ("balance",),
    ("lut_opt",),
    ("lut_opt_area",),
    ("rewrite", "balance"),
    ("rewrite_lp", "balance"),
    ("balance", "rewrite", "refactor"),
    ("balance", "rewrite_lpz", "lut_opt"),
    ("balance", "rewrite_lpz", "lut_opt_area"),
    ("balance", "refactor_lp", "rewrite_lp"),
    ("rewrite", "rewrite_zg", "balance", "refactor", "rewrite_zg"),
    ("balance", "rewrite", "balance", "refactor", "balance"),
    ("balance", "rewrite_lp", "refactor_lp", "rewrite_lpz"),
    ("lut_opt", "balance", "rewrite_lpz"),
    ("lut_opt_area", "balance", "rewrite_lpz"),
    ("rewrite_p12", "balance", "refactor_i12_c20"),
    ("rewrite_p12_lp", "balance", "refactor_i12_c20_lp"),
)


@dataclass(frozen=True)
class TargetQoR:
    area: int
    depth: int


def parse_ref_qor(input_aig: Path) -> TargetQoR | None:
    ref_path = input_aig.parent / "ref_qor.txt"
    if not ref_path.is_file():
        return None

    area = None
    depth = None
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("area:"):
            area = int(line.split(":", 1)[1].strip())
        elif line.startswith("level:"):
            depth = int(line.split(":", 1)[1].strip())

    if area is None or depth is None:
        return None
    return TargetQoR(area=area, depth=depth)


def result_rank(result: EvalResult, target: TargetQoR | None):
    if target is None:
        return (result.cost, result.depth, result.area)

    depth_gap = abs(result.depth - target.depth)
    area_gap = abs(result.area - target.area)
    exact = 0 if (depth_gap == 0 and area_gap == 0) else 1
    gap_score = 0.6 * depth_gap + 0.4 * area_gap
    return (exact, gap_score, depth_gap, area_gap, result.cost, result.depth, result.area)


def area_rank(result: EvalResult, target: TargetQoR | None):
    if target is None:
        return (result.area, result.depth, result.cost)
    area_gap = abs(result.area - target.area)
    depth_gap = abs(result.depth - target.depth)
    return (area_gap, depth_gap, result.cost, result.area, result.depth)


def depth_rank(result: EvalResult, target: TargetQoR | None):
    if target is None:
        return (result.depth, result.area, result.cost)
    depth_gap = abs(result.depth - target.depth)
    area_gap = abs(result.area - target.area)
    return (depth_gap, area_gap, result.cost, result.depth, result.area)


def is_exact_match(result: EvalResult, target: TargetQoR | None) -> bool:
    return target is not None and result.area == target.area and result.depth == target.depth


def prefix_key(steps: tuple[Action, ...]) -> tuple[str, ...]:
    return tuple(step.name for step in steps)


def instantiate_seed(template: tuple[str, ...]) -> tuple[Action, ...]:
    return tuple(ACTION_MAP[name] for name in template)


def seed_prefixes(max_steps: int) -> list[tuple[Action, ...]]:
    prefixes: list[tuple[Action, ...]] = [tuple()]
    for template in SEED_TEMPLATES:
        flow = instantiate_seed(template)
        for length in range(1, min(len(flow), max_steps) + 1):
            prefixes.append(flow[:length])
    return unique_prefixes(prefixes)


def choose_best_mapping(
    input_aig: Path,
    imap_bin: Path,
    steps: tuple[Action, ...],
    use_history: bool,
    timeout: float | None,
    target: TargetQoR | None,
    cache: dict[tuple[str, ...], EvalResult | None],
    evaluated_maps: list[int],
) -> EvalResult | None:
    key = prefix_key(steps)
    if key in cache:
        return cache[key]

    best = None
    for map_action in HEURISTIC_MAP_ACTIONS:
        candidate = Candidate(steps=steps, map_action=map_action)
        result = evaluate_candidate(
            aig_path=input_aig,
            imap_bin=imap_bin,
            candidate=candidate,
            use_history=use_history,
            timeout=timeout,
        )
        evaluated_maps[0] += 1
        if result is None:
            continue
        if best is None or result_rank(result, target) < result_rank(best, target):
            best = result
    cache[key] = best
    return best


def should_skip_action(prefix: tuple[Action, ...], action: Action) -> bool:
    if not prefix:
        return False

    if prefix[-1].name == action.name:
        return True

    if len(prefix) >= 2 and prefix[-2].name == action.name and prefix[-1].name.startswith("balance"):
        return True

    if len(prefix) >= 3 and tuple(step.name for step in prefix[-3:]) == (
        action.name,
        "balance",
        action.name,
    ):
        return True

    if prefix[-1].name.startswith("lut_opt") and action.name.startswith("lut_opt"):
        return True

    return False


def unique_prefixes(prefixes: list[tuple[Action, ...]]) -> list[tuple[Action, ...]]:
    seen = set()
    unique = []
    for prefix in prefixes:
        key = prefix_key(prefix)
        if key in seen:
            continue
        seen.add(key)
        unique.append(prefix)
    return unique


def select_diverse_results(
    results: list[EvalResult],
    target: TargetQoR | None,
    total_keep: int,
    per_archive: int,
) -> list[EvalResult]:
    if not results:
        return []

    selected: list[EvalResult] = []
    seen = set()

    def push(candidates):
        for result in candidates:
            key = prefix_key(result.candidate.steps)
            if key in seen:
                continue
            seen.add(key)
            selected.append(result)
            if len(selected) >= total_keep:
                return

    balanced = sorted(results, key=lambda r: result_rank(r, target))[:per_archive]
    area_first = sorted(results, key=lambda r: area_rank(r, target))[:per_archive]
    depth_first = sorted(results, key=lambda r: depth_rank(r, target))[:per_archive]

    push(balanced)
    if len(selected) < total_keep:
        push(area_first)
    if len(selected) < total_keep:
        push(depth_first)
    if len(selected) < total_keep:
        push(sorted(results, key=lambda r: result_rank(r, target)))
    return selected[:total_keep]


def queue_priority(
    result: EvalResult | None,
    target: TargetQoR | None,
    steps: tuple[Action, ...],
) -> tuple:
    if result is None:
        base = (10**9, 10**9, 10**9, 10**9, 10**9, 10**9)
    else:
        base = result_rank(result, target)
    return base + (len(steps),)


def push_queue(
    queue: list[tuple],
    queued: dict[tuple[str, ...], tuple],
    steps: tuple[Action, ...],
    priority: tuple,
    push_id: list[int],
) -> None:
    key = prefix_key(steps)
    old = queued.get(key)
    if old is not None and old <= priority:
        return
    queued[key] = priority
    heapq.heappush(queue, (priority, push_id[0], steps))
    push_id[0] += 1


def generate_expansions(prefix: tuple[Action, ...], max_steps: int) -> list[tuple[Action, ...]]:
    if len(prefix) >= max_steps:
        return []
    candidates: list[tuple[Action, ...]] = []
    for action in HEURISTIC_ACTIONS:
        if should_skip_action(prefix, action):
            continue
        candidates.append(prefix + (action,))
    return candidates


def generate_mutations(
    prefix: tuple[Action, ...],
    max_steps: int,
    budget: int,
) -> list[tuple[Action, ...]]:
    mutations: list[tuple[Action, ...]] = []
    seen = {prefix_key(prefix)}

    def add(candidate: tuple[Action, ...]) -> None:
        if len(mutations) >= budget:
            return
        if len(candidate) > max_steps:
            return
        key = prefix_key(candidate)
        if key in seen:
            return
        seen.add(key)
        mutations.append(candidate)

    # shrink
    for idx in range(len(prefix)):
        add(prefix[:idx] + prefix[idx + 1 :])

    # swap neighbors
    for idx in range(len(prefix) - 1):
        swapped = list(prefix)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        add(tuple(swapped))

    # append / prepend
    for action in FOCUSED_MUTATION_ACTIONS:
        add(prefix + (action,))
        add((action,) + prefix)

    # replace
    for idx in range(len(prefix)):
        for action in FOCUSED_MUTATION_ACTIONS:
            if prefix[idx].name == action.name:
                continue
            add(prefix[:idx] + (action,) + prefix[idx + 1 :])

    # insert
    for idx in range(len(prefix) + 1):
        for action in FOCUSED_MUTATION_ACTIONS:
            add(prefix[:idx] + (action,) + prefix[idx:])

    return mutations[:budget]


def write_sequence_file(path: Path, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sequence + "\n", encoding="utf-8")


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def maybe_commit_best(
    result: EvalResult | None,
    target: TargetQoR | None,
    best: EvalResult | None,
    quiet: bool,
    context: str,
) -> EvalResult | None:
    if result is None:
        return best
    if best is None or result_rank(result, target) < result_rank(best, target):
        if not quiet:
            print(
                f"best update [{context}]: cost={result.cost:.3f} logic_depth={result.depth} "
                f"area={result.area} labels={candidate_to_labels(result.candidate)}",
                file=sys.stderr,
            )
        return result
    return best


def finalize(
    output_seq: Path,
    input_aig: Path,
    target: TargetQoR | None,
    use_history: bool,
    max_steps: int,
    beam_width: int,
    local_rounds: int,
    mutation_budget: int,
    evaluated_prefixes: int,
    evaluated_maps: int,
    elapsed_sec: float,
    best: EvalResult | None,
    trace: list[dict],
) -> int:
    if best is None:
        fallback = "rewrite; balance; refactor; lut_opt; map_fpga;"
        write_sequence_file(output_seq, fallback)
        write_report(
            output_seq.with_suffix(output_seq.suffix + ".json"),
            {
                "input_aig": str(input_aig),
                "output_seq": str(output_seq),
                "max_steps": max_steps,
                "beam_width": beam_width,
                "local_rounds": local_rounds,
                "mutation_budget": mutation_budget,
                "use_history": use_history,
                "target": None if target is None else {"area": target.area, "depth": target.depth},
                "evaluated_prefixes": evaluated_prefixes,
                "evaluated_maps": evaluated_maps,
                "exact_match": False,
                "fallback": fallback,
                "elapsed_sec": elapsed_sec,
                "trace": trace,
            },
        )
        return 0

    best_sequence = serialize_sequence(build_sequence(best.candidate, use_history=use_history))
    write_sequence_file(output_seq, best_sequence)
    write_report(
        output_seq.with_suffix(output_seq.suffix + ".json"),
        {
            "input_aig": str(input_aig),
            "output_seq": str(output_seq),
            "max_steps": max_steps,
            "beam_width": beam_width,
            "local_rounds": local_rounds,
            "mutation_budget": mutation_budget,
            "use_history": use_history,
            "target": None if target is None else {"area": target.area, "depth": target.depth},
            "evaluated_prefixes": evaluated_prefixes,
            "evaluated_maps": evaluated_maps,
            "exact_match": is_exact_match(best, target),
            "best_cost": best.cost,
            "best_area": best.area,
            "best_depth": best.depth,
            "best_flow_labels": candidate_to_labels(best.candidate),
            "best_sequence": best_sequence,
            "elapsed_sec": elapsed_sec,
            "trace": trace,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heuristic target-aware search for the EDA23 iMAP problem."
    )
    parser.add_argument("input_aig", type=Path)
    parser.add_argument("output_seq", type=Path)
    parser.add_argument(
        "--imap-bin",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bin" / "imap",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("SEARCH_MAX_STEPS", os.environ.get("BF_MAX_STEPS", "5"))),
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=int(os.environ.get("SEARCH_BEAM_WIDTH", "8")),
        help="Elite archive size used by the search.",
    )
    parser.add_argument(
        "--local-rounds",
        type=int,
        default=int(os.environ.get("SEARCH_LOCAL_ROUNDS", "2")),
    )
    parser.add_argument(
        "--local-width",
        type=int,
        default=int(os.environ.get("SEARCH_LOCAL_WIDTH", "4")),
    )
    parser.add_argument(
        "--mutation-budget",
        type=int,
        default=int(os.environ.get("SEARCH_MUTATION_BUDGET", "12")),
    )
    parser.add_argument(
        "--restart-count",
        type=int,
        default=int(os.environ.get("SEARCH_RESTART_COUNT", "3")),
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=int(os.environ.get("SEARCH_MAX_EXPANSIONS", "180")),
        help="Maximum number of evaluated prefixes.",
    )
    parser.add_argument(
        "--tabu-size",
        type=int,
        default=int(os.environ.get("SEARCH_TABU_SIZE", "64")),
    )
    parser.add_argument(
        "--elite-interval",
        type=int,
        default=int(os.environ.get("SEARCH_ELITE_INTERVAL", "12")),
        help="Inject elite mutations into the queue every N expansions.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=(
            float(os.environ["SEARCH_EVAL_TIMEOUT"])
            if "SEARCH_EVAL_TIMEOUT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEARCH_SEED", "12345")),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
    )
    parser.add_argument(
        "--ignore-ref",
        action="store_true",
        help="Ignore sibling ref_qor.txt even if present.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    args = parser.parse_args()

    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    if args.beam_width <= 0:
        raise SystemExit("--beam-width must be positive")
    if args.local_rounds < 0 or args.local_width <= 0 or args.mutation_budget <= 0:
        raise SystemExit("local search arguments must be positive")
    if args.max_expansions <= 0 or args.tabu_size <= 0 or args.elite_interval <= 0:
        raise SystemExit("queue search arguments must be positive")

    input_aig = args.input_aig.resolve()
    output_seq = args.output_seq.resolve()
    imap_bin = args.imap_bin.resolve()
    use_history = not args.no_history
    target = None if args.ignore_ref else parse_ref_qor(input_aig)

    if not input_aig.is_file():
        raise SystemExit(f"input AIG not found: {input_aig}")
    if not imap_bin.is_file():
        raise SystemExit(f"imap binary not found: {imap_bin}")

    if not args.quiet:
        if target is None:
            print(
                f"heuristic search: {input_aig.name}, no ref target, optimizing cost directly",
                file=sys.stderr,
            )
        else:
            print(
                f"heuristic search: {input_aig.name}, target area={target.area} depth={target.depth}",
                file=sys.stderr,
            )

    start = time.time()
    rng = random.Random(args.seed)
    evaluated_prefixes = 0
    evaluated_maps = [0]
    cache: dict[tuple[str, ...], EvalResult | None] = {}
    best: EvalResult | None = None
    trace: list[dict] = []
    expanded: set[tuple[str, ...]] = set()
    tabu_queue: deque[tuple[str, ...]] = deque()
    tabu_set: set[tuple[str, ...]] = set()
    global_results: list[EvalResult] = []
    elite_prefixes: list[tuple[Action, ...]] = []
    queue: list[tuple] = []
    queued: dict[tuple[str, ...], tuple] = {}
    push_id = [0]

    for seed in seed_prefixes(args.max_steps):
        push_queue(queue, queued, seed, (0, 0, 0), push_id)

    while queue and evaluated_prefixes < args.max_expansions:
        priority, _, steps = heapq.heappop(queue)
        key = prefix_key(steps)
        if key in expanded:
            continue
        if queued.get(key) != priority:
            continue

        expanded.add(key)
        tabu_queue.append(key)
        tabu_set.add(key)
        if len(tabu_queue) > args.tabu_size:
            old = tabu_queue.popleft()
            tabu_set.discard(old)

        result = choose_best_mapping(
            input_aig=input_aig,
            imap_bin=imap_bin,
            steps=steps,
            use_history=use_history,
            timeout=args.timeout,
            target=target,
            cache=cache,
            evaluated_maps=evaluated_maps,
        )
        evaluated_prefixes += 1
        best = maybe_commit_best(result, target, best, args.quiet, f"bestfirst:{len(steps)}")
        if result is not None:
            global_results.append(result)
            elite = select_diverse_results(
                global_results,
                target=target,
                total_keep=args.beam_width,
                per_archive=max(2, args.beam_width // 2),
            )
            elite_prefixes = [r.candidate.steps for r in elite]
            trace.append(
                {
                    "phase": "best_first",
                    "expansion": evaluated_prefixes,
                    "expanded_steps": [step.name for step in steps],
                    "logic_depth": result.depth,
                    "area": result.area,
                    "cost": result.cost,
                }
            )
            if is_exact_match(result, target):
                return finalize(
                    output_seq=output_seq,
                    input_aig=input_aig,
                    target=target,
                    use_history=use_history,
                    max_steps=args.max_steps,
                    beam_width=args.beam_width,
                    local_rounds=args.local_rounds,
                    mutation_budget=args.mutation_budget,
                    evaluated_prefixes=evaluated_prefixes,
                    evaluated_maps=evaluated_maps[0],
                    elapsed_sec=time.time() - start,
                    best=result,
                    trace=trace,
                )

        current_priority = queue_priority(result, target, steps)
        for child in generate_expansions(steps, args.max_steps):
            child_key = prefix_key(child)
            if child_key in expanded or child_key in tabu_set:
                continue
            push_queue(queue, queued, child, current_priority, push_id)

        if result is not None and steps:
            for mutated in generate_mutations(steps, args.max_steps, min(4, args.mutation_budget)):
                mutated_key = prefix_key(mutated)
                if mutated_key in expanded or mutated_key in tabu_set:
                    continue
                push_queue(queue, queued, mutated, current_priority, push_id)

        if evaluated_prefixes % args.elite_interval == 0 and elite_prefixes:
            trace.append(
                {
                    "phase": "elite_injection",
                    "expansion": evaluated_prefixes,
                    "elite_prefixes": [list(prefix_key(p)) for p in elite_prefixes],
                }
            )
            for elite_prefix in elite_prefixes[: args.local_width]:
                for mutation in generate_mutations(elite_prefix, args.max_steps, args.mutation_budget):
                    mutation_key = prefix_key(mutation)
                    if mutation_key in expanded or mutation_key in tabu_set:
                        continue
                    cached = cache.get(mutation_key)
                    push_queue(
                        queue,
                        queued,
                        mutation,
                        queue_priority(cached, target, mutation),
                        push_id,
                    )
                for child in generate_expansions(elite_prefix, args.max_steps):
                    child_key = prefix_key(child)
                    if child_key in expanded or child_key in tabu_set:
                        continue
                    cached = cache.get(child_key)
                    push_queue(
                        queue,
                        queued,
                        child,
                        queue_priority(cached, target, child),
                        push_id,
                    )

    ranked_cache = [r for r in cache.values() if r is not None]
    ranked_cache.sort(key=lambda r: result_rank(r, target))
    local_frontier = unique_prefixes([r.candidate.steps for r in ranked_cache[: args.local_width]])

    restart_pool = unique_prefixes(elite_prefixes)
    if restart_pool:
        rng.shuffle(restart_pool)
        local_frontier.extend(restart_pool[: args.restart_count])
        local_frontier = unique_prefixes(local_frontier)

    for round_index in range(args.local_rounds):
        round_results: list[EvalResult] = []
        next_local_frontier: list[tuple[Action, ...]] = []

        for prefix in local_frontier[: args.local_width]:
            for mutation in generate_mutations(prefix, args.max_steps, args.mutation_budget):
                result = choose_best_mapping(
                    input_aig=input_aig,
                    imap_bin=imap_bin,
                    steps=mutation,
                    use_history=use_history,
                    timeout=args.timeout,
                    target=target,
                    cache=cache,
                    evaluated_maps=evaluated_maps,
                )
                evaluated_prefixes += 1
                best = maybe_commit_best(result, target, best, args.quiet, f"tabu-local:r{round_index}")
                if result is None:
                    continue
                round_results.append(result)

                if is_exact_match(result, target):
                    return finalize(
                        output_seq=output_seq,
                        input_aig=input_aig,
                        target=target,
                        use_history=use_history,
                        max_steps=args.max_steps,
                        beam_width=args.beam_width,
                        local_rounds=args.local_rounds,
                        mutation_budget=args.mutation_budget,
                        evaluated_prefixes=evaluated_prefixes,
                        evaluated_maps=evaluated_maps[0],
                        elapsed_sec=time.time() - start,
                        best=result,
                        trace=trace,
                    )

            current = choose_best_mapping(
                input_aig=input_aig,
                imap_bin=imap_bin,
                steps=prefix,
                use_history=use_history,
                timeout=args.timeout,
                target=target,
                cache=cache,
                evaluated_maps=evaluated_maps,
            )
            if current is not None:
                round_results.append(current)

        if not round_results:
            break

        kept = select_diverse_results(
            round_results,
            target=target,
            total_keep=args.local_width,
            per_archive=max(2, args.local_width),
        )
        next_local_frontier = unique_prefixes([r.candidate.steps for r in kept])
        trace.append(
            {
                "phase": "tabu_local",
                "round": round_index,
                "kept_prefixes": [
                    {
                        "steps": [step.name for step in result.candidate.steps],
                        "logic_depth": result.depth,
                        "area": result.area,
                        "cost": result.cost,
                    }
                    for result in kept
                ],
            }
        )
        local_frontier = next_local_frontier

    if not args.quiet and best is not None:
        print(
            f"done: best cost={best.cost:.3f} logic_depth={best.depth} area={best.area} "
            f"evaluated_prefixes={evaluated_prefixes} evaluated_maps={evaluated_maps[0]}",
            file=sys.stderr,
        )

    return finalize(
        output_seq=output_seq,
        input_aig=input_aig,
        target=target,
        use_history=use_history,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        local_rounds=args.local_rounds,
        mutation_budget=args.mutation_budget,
        evaluated_prefixes=evaluated_prefixes,
        evaluated_maps=evaluated_maps[0],
        elapsed_sec=time.time() - start,
        best=best,
        trace=trace,
    )


if __name__ == "__main__":
    raise SystemExit(main())
