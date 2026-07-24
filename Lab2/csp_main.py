
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

from gas_world import build_world, PEAK_SLOTS, MODERATE_SLOTS
from csp_model import (
    ALL_CONSTRAINTS,
    CSP,
    Policy,
    build_csp,
    default_enabled,
    household_pairs,
    print_domain_summary,
)
from backtracking import backtracking_search
from min_conflicts import min_conflicts
from visualize import (
    plot_assignment_map,
    plot_comparison_bars,
    plot_convergence,
    plot_station_loads,
)


# ---------------------------------------------------------------------------
# Lab 1's CLI helpers (carried over verbatim)
# ---------------------------------------------------------------------------

def _ask_bounded_float(prompt, default, low, high):
    while True:
        raw = input(f"{prompt} [{default}] (range: {low} to {high}): ").strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("[input] Please enter a numeric value.")
            continue
        if low <= value <= high:
            return value
        print(f"[input] Value must be between {low} and {high}.")


def _ask_int(prompt, default, low, high):
    while True:
        raw = input(f"{prompt} [{default}] (range: {low} to {high}): ").strip()
        if not raw:
            return int(default)
        try:
            value = int(raw)
        except ValueError:
            print("[input] Please enter an integer value.")
            continue
        if low <= value <= high:
            return value
        print(f"[input] Value must be between {low} and {high}.")


def _ask_menu(prompt, options: dict[str, str], default_key: str) -> str:
    print(prompt)
    for key, label in options.items():
        marker = " (default)" if key == default_key else ""
        print(f"  {key}. {label}{marker}")
    while True:
        raw = input("Choose option number: ").strip()
        if not raw:
            return options[default_key]
        if raw in options:
            return options[raw]
        print(f"[input] Invalid choice. Choose one of: {', '.join(options.keys())}")


# ---------------------------------------------------------------------------
# bbox presets — keep things tractable
# ---------------------------------------------------------------------------

BBOX_PRESETS: dict[str, tuple[float, float, float, float] | None] = {
    "dhanmondi":   (23.735, 23.770, 90.360, 90.395),
    "mohakhali":   (23.770, 23.810, 90.395, 90.430),
    "mirpur":      (23.795, 23.835, 90.345, 90.385),
    "old_dhaka":   (23.700, 23.730, 90.395, 90.430),
    "full_dhaka":  None,
}


def _ask_bbox() -> tuple[str, tuple[float, float, float, float] | None]:
    options = {str(i + 1): name for i, name in enumerate(BBOX_PRESETS.keys())}
    name = _ask_menu(
        "\nArea preset (bbox for users + stations):",
        options,
        default_key="1",
    )
    return name, BBOX_PRESETS[name]


# ---------------------------------------------------------------------------
# constraint toggles
# ---------------------------------------------------------------------------

def _ask_constraints(enabled: dict[str, bool]) -> dict[str, bool]:
    print("\n=== Active constraints (toggle by number; blank=keep, 'all on', 'all off') ===")
    while True:
        for i, name in enumerate(ALL_CONSTRAINTS, 1):
            status = "ON " if enabled[name] else "OFF"
            print(f"  {i:2d}. [{status}] {name}")
        raw = input("Toggle which (e.g. '3' or '3,5,7') or Enter to continue: ").strip().lower()
        if not raw:
            return enabled
        if raw in ("all on", "on"):
            enabled = {n: True for n in ALL_CONSTRAINTS}
            continue
        if raw in ("all off", "off"):
            enabled = {n: False for n in ALL_CONSTRAINTS}
            continue
        try:
            indices = [int(x.strip()) for x in raw.replace(",", " ").split()]
        except ValueError:
            print("[input] Use numbers separated by commas/spaces.")
            continue
        for idx in indices:
            if 1 <= idx <= len(ALL_CONSTRAINTS):
                name = ALL_CONSTRAINTS[idx - 1]
                enabled[name] = not enabled[name]
            else:
                print(f"[input] No constraint #{idx}.")


def _ask_policy(default: Policy) -> Policy:
    print("\n=== Policy knobs ===")
    parity = _ask_menu(
        "Plate-parity rationing today:",
        {"1": "any", "2": "odd", "3": "even"},
        default_key="1",
    )
    public_frac = _ask_bounded_float(
        "G3' public-transport reserve fraction",
        default.public_reserve_fraction, 0.0, 0.95,
    )
    strategic_frac = _ask_bounded_float(
        "G4 strategic reserve fraction",
        default.strategic_reserve_fraction, 0.0, 0.5,
    )
    queue_cap = _ask_int(
        "G8 queue cap (max users per station)",
        default.queue_cap, 1, 200,
    )
    min_quota = _ask_int(
        "G9 minimum-service quota (each station must serve >=)",
        default.min_service_quota, 0, 20,
    )
    hh_frac = _ask_bounded_float(
        "B1 household pairing fraction (0=no households)",
        default.household_fraction, 0.0, 0.6,
    )
    return Policy(
        today_parity=parity,
        public_reserve_fraction=public_frac,
        strategic_reserve_fraction=strategic_frac,
        queue_cap=queue_cap,
        min_service_quota=min_quota,
        household_fraction=hh_frac,
        households_per_group=2,
    )


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def _log_run(run_id: str, scenario_info: dict, solver: str, result_info: dict,
             csv_path: str = "cache/csp_history.csv") -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_id", "solver", "bbox", "n_users", "time_slot", "seed",
        "parity", "queue_cap", "min_quota",
        "found", "cost", "served", "unassignable",
        "expanded", "backtracks", "iters", "violations",
        "runtime_ms",
    ]
    row = {
        "run_id": run_id,
        "solver": solver,
        **{k: scenario_info.get(k) for k in [
            "bbox", "n_users", "time_slot", "seed", "parity", "queue_cap", "min_quota"
        ]},
        **result_info,
    }
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# pretty printers
# ---------------------------------------------------------------------------

def _print_assignment(world, assignment: dict[int, int], unassignable: list[int]) -> None:
    print(f"\n  Served: {len(assignment)}/{len(world.users)}, "
          f"Infeasible: {len(unassignable)}")
    if not assignment:
        return
    print(f"  {'user':>5s} | {'class':9s} | {'plate':5s} | "
          f"{'zone':5s} | {'-> station':5s} | {'name':25s} | {'meters':>7s}")
    print("  " + "-" * 80)
    for uid in sorted(assignment):
        sid = assignment[uid]
        u = world.users[uid]
        s = world.stations[sid]
        d = world.distance(u, s)
        print(f"  {uid:5d} | {u.vehicle_class:9s} | {u.plate_parity:5s} | "
              f"{u.zone:5s} | -> S{sid:<2d}   | {s.name[:25]:25s} | {d:7.0f}")


def _print_three_way_comparison(world, naive_bt, heur_bt, mc_result) -> None:
    """Three solvers on the same scenario, side-by-side.

    Compares: Naive BT (no heuristics) | Heuristic BT (MRV+Degree+LCV+FC)
    | Min-Conflicts (local search).
    """
    print("\n  === Three-way comparison on the SAME scenario ===")
    print("  (same world, same constraints, same seed — only the algorithm differs)")

    headers = ["Metric",            "Naive BT",       "Heuristic BT",   "Min-Conflicts"]
    print(f"  {headers[0]:<26s} | {headers[1]:>14s} | {headers[2]:>14s} | {headers[3]:>14s}")
    print("  " + "-" * 78)
    def fmt(val, kind="num"):
        if val is None: return "—"
        if kind == "bool": return "yes" if val else "no"
        if kind == "cost": return f"{val:.0f}" if math.isfinite(val) else "inf"
        if kind == "ms":   return f"{val:.1f}"
        return f"{val}"

    rows = [
        ("Found a solution",
         fmt(naive_bt.found, "bool"), fmt(heur_bt.found, "bool"), fmt(mc_result.found, "bool")),
        ("Total cost (lower=better)",
         fmt(naive_bt.cost, "cost"), fmt(heur_bt.cost, "cost"), fmt(mc_result.cost, "cost")),
        ("Users served",
         fmt(len(naive_bt.assignment)), fmt(len(heur_bt.assignment)), fmt(len(mc_result.assignment))),
        ("Runtime (ms)",
         fmt(naive_bt.runtime_ms, "ms"), fmt(heur_bt.runtime_ms, "ms"), fmt(mc_result.runtime_ms, "ms")),
        ("Search nodes expanded",
         fmt(naive_bt.nodes_expanded), fmt(heur_bt.nodes_expanded), "n/a (local search)"),
        ("Backtracks",
         fmt(naive_bt.backtracks), fmt(heur_bt.backtracks), "n/a"),
        ("MC iterations",
         "n/a", "n/a", fmt(mc_result.iterations)),
        ("Final violations (MC)",
         "n/a", "n/a", fmt(mc_result.final_violations)),
        ("Timed out?",
         fmt(naive_bt.timed_out, "bool"), fmt(heur_bt.timed_out, "bool"), fmt(mc_result.timed_out, "bool")),
    ]
    for r in rows:
        print(f"  {r[0]:<26s} | {r[1]:>14s} | {r[2]:>14s} | {r[3]:>14s}")

    # quick narrative
    if naive_bt.nodes_expanded > 0 and heur_bt.nodes_expanded > 0:
        speedup = naive_bt.nodes_expanded / max(1, heur_bt.nodes_expanded)
        print(f"\n  Heuristics pruned the search tree by ~{speedup:.1f}× "
              f"(naive expanded {naive_bt.nodes_expanded}, heuristic expanded {heur_bt.nodes_expanded}).")
    if naive_bt.timed_out and not heur_bt.timed_out:
        print("   Naive BT timed out while heuristic BT finished ")
    if mc_result.found and (naive_bt.timed_out or not naive_bt.found):
        print("  Min-Conflicts scaled where Naive BT couldn't")


def _print_method_comparison(world, bt_result, mc_result) -> None:
    """Side-by-side: where did each method send each user?"""
    print("\n  === Side-by-side comparison: Method 1 vs Method 2 ===")
    all_uids = sorted(set(bt_result.assignment) | set(mc_result.assignment))
    if not all_uids:
        print("  (neither method assigned anyone)")
        return
    print(f"  {'user':>4s} | {'class':9s} | "
          f"{'M1 sta':>7s} {'M1 name':22s} {'M1 m':>6s} | "
          f"{'M2 sta':>7s} {'M2 name':22s} {'M2 m':>6s} | match")
    print("  " + "-" * 110)
    same = diff = only_m1 = only_m2 = 0
    for uid in all_uids:
        u = world.users[uid]
        m1 = bt_result.assignment.get(uid)
        m2 = mc_result.assignment.get(uid)

        def _fmt(sid):
            if sid is None:
                return ("    —", " " * 22, "     —")
            s = world.stations[sid]
            return (f"   S{sid:<2d}", f"{s.name[:22]:22s}", f"{world.distance(u, s):6.0f}")

        m1_cells = _fmt(m1)
        m2_cells = _fmt(m2)

        if m1 is None and m2 is not None:
            tag = "MC only"; only_m2 += 1
        elif m2 is None and m1 is not None:
            tag = "BT only"; only_m1 += 1
        elif m1 == m2:
            tag = "✓ same "; same += 1
        else:
            tag = "✗ diff "; diff += 1

        print(f"  {uid:4d} | {u.vehicle_class:9s} | "
              f"{m1_cells[0]} {m1_cells[1]} {m1_cells[2]} | "
              f"{m2_cells[0]} {m2_cells[1]} {m2_cells[2]} | {tag}")

    print("  " + "-" * 110)
    print(f"  Totals:  both agree (same station)={same}  | "
          f"both serve, different stations={diff}  | "
          f"only BT={only_m1}  | only MC={only_m2}")
    print(f"  BT cost = {bt_result.cost:.0f}  vs  MC cost = {mc_result.cost:.0f}  "
          f"(diff = {mc_result.cost - bt_result.cost:+.0f})")
    print(f"  BT runtime = {bt_result.runtime_ms:.1f} ms  vs  "
          f"MC runtime = {mc_result.runtime_ms:.1f} ms")


# ---------------------------------------------------------------------------
# the main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Lab 2 — CSP Gas Allocation in Dhaka")
    print("  (reuses Lab 1's Dhaka OSM graph + distance machinery)")
    print("=" * 72)

    enabled = default_enabled()
    policy = Policy()
    world = None
    world_signature = None

    while True:
        print("\n=== NEW SCENARIO ===")
        n_users = _ask_int("Number of users (variables)", 15, 2, 500)
        time_slot = _ask_menu(
            "\nTime slot (drives jam multiplier + G6' CNG-peak rule):",
            {
                "1": "07:00-10:00",
                "2": "10:00-12:00",
                "3": "15:00-17:00",
                "4": "17:00-21:00",
                "5": "21:00-06:00",
            },
            default_key="2",
        )
        seed = _ask_int("Random seed", 42, 0, 999_999)
        bbox_name, bbox = _ask_bbox()
        max_stations = _ask_int(
            "Max stations (subsample after bbox filter)",
            10, 2, 100,
        )

        signature = (n_users, time_slot, seed, bbox_name, max_stations)
        if signature != world_signature:
            print("\n[main] Building world (this rebuilds the distance oracle)...")
            world = build_world(
                n_users=n_users,
                time_slot=time_slot,
                seed=seed,
                bbox=bbox,
                max_stations=max_stations,
                area_name=bbox_name,
            )
            world_signature = signature
        else:
            print("\n[main] Reusing existing world — no oracle rebuild.")

        # constraints + policy
        enabled = _ask_constraints(enabled)
        policy = _ask_policy(policy)

        csp = build_csp(world, policy=policy, enabled=enabled, household_seed=seed)
        print_domain_summary(csp)

        # solver choice — 2 methods aligned with the slide deck:
        #   Method 1: Backtracking with all 3 heuristics (MRV + Degree + LCV) + FC
        #   Method 2: Min-Conflicts local search heuristic
        which = _ask_menu(
            "\nSolver (Methods):",
            {
                "1": "Both methods (Method 1 + Method 2)",
                "2": "Method 1 only — Backtracking (MRV + Degree + LCV + FC)",
                "3": "Method 2 only — Min-Conflicts (local search)",
                "4": "Heuristic-ablation study — BT plain vs +MRV vs +MRV+LCV vs +MRV+LCV+FC",
                "5": "Three-way compare — Naive BT + Heuristic BT + Min-Conflicts (Recommended)",
            },
            default_key="5",
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"cache/plots/{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        scenario_info = dict(
            bbox=bbox_name, n_users=n_users, time_slot=time_slot, seed=seed,
            parity=policy.today_parity, queue_cap=policy.queue_cap,
            min_quota=policy.min_service_quota,
        )

        bt_result = mc_result = naive_bt_result = None
        bt_variants = []

        run_three_way = which.startswith("Three-way")
        run_method1 = (which.startswith("Both") or which.startswith("Method 1")
                       or run_three_way)
        run_method2 = (which.startswith("Both") or which.startswith("Method 2")
                       or run_three_way)
        run_ablation = which.startswith("Heuristic-ablation")

        if run_three_way:
            print("\n[main] Naive Backtracking (no heuristics) — baseline")
            print("       (no MRV / Degree / LCV / FC)")
            naive_bt_result = backtracking_search(
                csp, use_mrv=False, use_lcv=False, use_fc=False, time_limit_ms=30_000,
            )
            print(f"  {naive_bt_result.summary()}")
            plot_assignment_map(
                world, naive_bt_result.assignment, naive_bt_result.unassignable,
                title=f"Naive BT (no heuristics) | cost={naive_bt_result.cost:.0f} | "
                      f"served={len(naive_bt_result.assignment)}/{n_users}",
                output_path=str(out_dir / "naive_bt_map.png"), bbox=bbox,
            )
            _log_run(run_id, scenario_info, "naive_BT_no_heuristics", dict(
                found=naive_bt_result.found, cost=naive_bt_result.cost,
                served=len(naive_bt_result.assignment),
                unassignable=len(naive_bt_result.unassignable),
                expanded=naive_bt_result.nodes_expanded,
                backtracks=naive_bt_result.backtracks,
                iters="", violations="", runtime_ms=naive_bt_result.runtime_ms,
            ))

        if run_method1:
            print("\n[main] Method 1: Backtracking + MRV + Degree + LCV + Forward Checking")
            print("       (heuristics H1+H2+H3 + inference)")
            bt_result = backtracking_search(csp, use_mrv=True, use_lcv=True, use_fc=True,
                                            time_limit_ms=30_000)
            print(f"  {bt_result.summary()}")
            _print_assignment(world, bt_result.assignment, bt_result.unassignable)
            plot_assignment_map(
                world, bt_result.assignment, bt_result.unassignable,
                title=f"Method 1 (BT+MRV+Degree+LCV+FC) | cost={bt_result.cost:.0f} | served={len(bt_result.assignment)}/{n_users}",
                output_path=str(out_dir / "method1_bt_map.png"),
                bbox=bbox,
            )
            plot_station_loads(
                world, bt_result.assignment,
                policy_strategic_frac=policy.strategic_reserve_fraction,
                policy_public_frac=policy.public_reserve_fraction,
                output_path=str(out_dir / "method1_bt_loads.png"),
            )
            _log_run(run_id, scenario_info, "method1_BT_MRV_Degree_LCV_FC", dict(
                found=bt_result.found, cost=bt_result.cost,
                served=len(bt_result.assignment), unassignable=len(bt_result.unassignable),
                expanded=bt_result.nodes_expanded, backtracks=bt_result.backtracks,
                iters="", violations="", runtime_ms=bt_result.runtime_ms,
            ))

        if run_method2:
            print("\n[main] Method 2: Min-Conflicts (local search heuristic)")
            mc_result = min_conflicts(csp, max_steps=5000, max_restarts=5, seed=seed,
                                      time_limit_ms=30_000)
            print(f"  {mc_result.summary()}")
            _print_assignment(world, mc_result.assignment, mc_result.unassignable)
            plot_assignment_map(
                world, mc_result.assignment, mc_result.unassignable,
                title=f"Method 2 (Min-Conflicts) | cost={mc_result.cost:.0f} | served={len(mc_result.assignment)}/{n_users}",
                output_path=str(out_dir / "method2_mc_map.png"),
                bbox=bbox,
            )
            plot_station_loads(
                world, mc_result.assignment,
                policy_strategic_frac=policy.strategic_reserve_fraction,
                policy_public_frac=policy.public_reserve_fraction,
                output_path=str(out_dir / "method2_mc_loads.png"),
            )
            plot_convergence(mc_result.convergence, output_path=str(out_dir / "method2_mc_convergence.png"))
            _log_run(run_id, scenario_info, "method2_MinConflicts", dict(
                found=mc_result.found, cost=mc_result.cost,
                served=len(mc_result.assignment), unassignable=len(mc_result.unassignable),
                expanded="", backtracks="",
                iters=mc_result.iterations, violations=mc_result.final_violations,
                runtime_ms=mc_result.runtime_ms,
            ))

        if run_ablation:
            print("\n[main] Heuristic ablation: which heuristic helps how much?")
            cost_by, runtime_by = {}, {}
            for label, (mrv, lcv, fc) in [
                ("plain BT", (False, False, False)),
                ("+ MRV+Degree", (True, False, False)),
                ("+ MRV+Degree+LCV", (True, True, False)),
                ("+ MRV+Degree+LCV+FC", (True, True, True)),
            ]:
                r = backtracking_search(csp, use_mrv=mrv, use_lcv=lcv, use_fc=fc,
                                        time_limit_ms=30_000)
                bt_variants.append((label, r))
                cost_by[label] = r.cost if r.found else 0.0
                runtime_by[label] = r.runtime_ms
                print(f"  {label:24s} {r.summary()}")
                _log_run(run_id, scenario_info, f"ablation_{label.replace(' ', '_')}", dict(
                    found=r.found, cost=r.cost,
                    served=len(r.assignment), unassignable=len(r.unassignable),
                    expanded=r.nodes_expanded, backtracks=r.backtracks,
                    iters="", violations="", runtime_ms=r.runtime_ms,
                ))
            plot_comparison_bars(
                cost_by, "Heuristic ablation — total cost", "Cost",
                output_path=str(out_dir / "ablation_cost.png"),
            )
            plot_comparison_bars(
                runtime_by, "Heuristic ablation — runtime", "ms",
                output_path=str(out_dir / "ablation_runtime.png"),
            )

        if run_three_way and naive_bt_result and bt_result and mc_result:
            _print_three_way_comparison(world, naive_bt_result, bt_result, mc_result)
            plot_comparison_bars(
                {"Naive BT": naive_bt_result.cost,
                 "Heuristic BT": bt_result.cost,
                 "Min-Conflicts": mc_result.cost},
                "Three-way — total cost (lower is better)", "Cost",
                output_path=str(out_dir / "three_way_cost.png"),
            )
            plot_comparison_bars(
                {"Naive BT": naive_bt_result.runtime_ms,
                 "Heuristic BT": bt_result.runtime_ms,
                 "Min-Conflicts": mc_result.runtime_ms},
                "Three-way — runtime", "ms",
                output_path=str(out_dir / "three_way_runtime.png"),
            )
            # search effort (MC has no expanded count → use 0 as placeholder)
            plot_comparison_bars(
                {"Naive BT": naive_bt_result.nodes_expanded,
                 "Heuristic BT": bt_result.nodes_expanded,
                 "Min-Conflicts": mc_result.iterations},
                "Three-way — search effort (BT: expanded nodes, MC: iterations)",
                "count",
                output_path=str(out_dir / "three_way_effort.png"),
            )
        elif bt_result and mc_result:
            _print_method_comparison(world, bt_result, mc_result)
            plot_comparison_bars(
                {"Method 1 (BT+heuristics)": bt_result.cost,
                 "Method 2 (Min-Conflicts)": mc_result.cost},
                "Total cost (lower is better)", "Cost",
                output_path=str(out_dir / "method_cost_compare.png"),
            )
            plot_comparison_bars(
                {"Method 1 (BT+heuristics)": bt_result.runtime_ms,
                 "Method 2 (Min-Conflicts)": mc_result.runtime_ms},
                "Runtime", "ms",
                output_path=str(out_dir / "method_runtime_compare.png"),
            )

        print(f"\n[main] Plots & log: {out_dir}")
        again = _ask_menu(
            "\nAnother run?",
            {"1": "yes — same world, different settings (fast)",
             "2": "yes — new world (rebuild)",
             "3": "no — quit"},
            default_key="1",
        )
        if again.startswith("no"):
            print("\nFinished.")
            break
        if again.startswith("yes — new world"):
            world_signature = None  # force rebuild


if __name__ == "__main__":
    main()
