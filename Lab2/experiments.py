"""
experiments.py — Run A/B/C automated batch for the report
=========================================================

Run A (head-to-head) — same small instance, BT vs Min-Conflicts.
Run B (scaling sweep) — fix bbox + stations, vary N users, watch where BT
    times out and Min-Conflicts keeps going.
Run C (realistic large) — full Dhaka graph, many users + stations,
    Min-Conflicts only.

Each function writes CSVs + figures into cache/experiments/<run>/ so they can
be dropped straight into the report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import csv
import statistics
import time

from gas_world import build_world
from csp_model import build_csp, Policy, default_enabled
from backtracking import backtracking_search
from min_conflicts import min_conflicts
from visualize import (
    plot_assignment_map,
    plot_comparison_bars,
    plot_convergence,
    plot_scaling,
    plot_station_loads,
)


OUTPUT_ROOT = Path("cache/experiments")
DHANMONDI = (23.735, 23.770, 90.360, 90.395)


# ---------------------------------------------------------------------------
# Run A — head-to-head on a small instance
# ---------------------------------------------------------------------------

def run_A(n_users: int = 12, seed: int = 42, time_slot: str = "10:00-12:00") -> dict:
    out = OUTPUT_ROOT / "runA_head_to_head"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[Run A] n_users={n_users}, bbox=Dhanmondi, time_slot={time_slot}")

    world = build_world(
        n_users=n_users, time_slot=time_slot, seed=seed,
        bbox=DHANMONDI, max_stations=6, area_name="dhanmondi",
    )
    policy = Policy(today_parity="any", min_service_quota=0)
    csp = build_csp(world, policy=policy)

    bt = backtracking_search(csp, use_mrv=True, use_lcv=True, use_fc=True, time_limit_ms=30_000)
    mc = min_conflicts(csp, max_steps=5000, max_restarts=5, seed=seed, time_limit_ms=30_000)

    print(f"  BT: {bt.summary()}")
    print(f"  MC: {mc.summary()}")

    plot_assignment_map(
        world, bt.assignment, bt.unassignable,
        title=f"Run A — Backtracking | cost={bt.cost:.0f} | served={len(bt.assignment)}/{n_users}",
        output_path=str(out / "A_bt_map.png"), bbox=DHANMONDI,
    )
    plot_assignment_map(
        world, mc.assignment, mc.unassignable,
        title=f"Run A — Min-Conflicts | cost={mc.cost:.0f} | served={len(mc.assignment)}/{n_users}",
        output_path=str(out / "A_mc_map.png"), bbox=DHANMONDI,
    )
    plot_station_loads(world, bt.assignment, output_path=str(out / "A_bt_loads.png"))
    plot_station_loads(world, mc.assignment, output_path=str(out / "A_mc_loads.png"))
    plot_convergence(mc.convergence, output_path=str(out / "A_mc_convergence.png"))
    plot_comparison_bars(
        {"Backtracking": bt.cost, "Min-Conflicts": mc.cost},
        "Run A — Total cost", "Cost",
        output_path=str(out / "A_cost_compare.png"),
    )
    plot_comparison_bars(
        {"Backtracking": bt.runtime_ms, "Min-Conflicts": mc.runtime_ms},
        "Run A — Runtime", "ms",
        output_path=str(out / "A_runtime_compare.png"),
    )

    # CSV row
    with (out / "A_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "found", "cost", "served", "unassignable",
                    "runtime_ms", "expanded_or_iters", "backtracks_or_violations"])
        w.writerow(["Backtracking", bt.found, f"{bt.cost:.1f}", len(bt.assignment),
                    len(bt.unassignable), f"{bt.runtime_ms:.2f}",
                    bt.nodes_expanded, bt.backtracks])
        w.writerow(["Min-Conflicts", mc.found, f"{mc.cost:.1f}", len(mc.assignment),
                    len(mc.unassignable), f"{mc.runtime_ms:.2f}",
                    mc.iterations, mc.final_violations])

    print(f"  -> {out}")
    return {"bt": bt, "mc": mc, "out": str(out)}


# ---------------------------------------------------------------------------
# Run B — scaling sweep
# ---------------------------------------------------------------------------

def run_B(
    sizes: list[int] | None = None,
    bt_timeout_ms: float = 30_000,
    mc_timeout_ms: float = 30_000,
    seed: int = 42,
    time_slot: str = "10:00-12:00",
) -> dict:
    out = OUTPUT_ROOT / "runB_scaling"
    out.mkdir(parents=True, exist_ok=True)
    sizes = sizes or [10, 20, 40, 80, 160]
    print(f"\n[Run B] Scaling sweep N={sizes}")

    # We use a larger bbox for Run B so growing N has room to spread.
    bbox = (23.700, 23.830, 90.345, 90.430)

    bt_runtimes: list[float] = []
    mc_runtimes: list[float] = []
    bt_found: list[bool] = []
    mc_found: list[bool] = []
    bt_costs: list[float] = []
    mc_costs: list[float] = []

    for n in sizes:
        print(f"\n  -- N={n} --")
        world = build_world(
            n_users=n, time_slot=time_slot, seed=seed,
            bbox=bbox, max_stations=min(30, max(8, n // 4)),
            area_name="full_dhaka",  # run B uses a large multi-area bbox → 3×3 grid
        )
        policy = Policy(today_parity="any", min_service_quota=0)
        csp = build_csp(world, policy=policy)

        bt = backtracking_search(csp, use_mrv=True, use_lcv=True, use_fc=True,
                                 time_limit_ms=bt_timeout_ms)
        print(f"    BT: {bt.summary()}")
        bt_runtimes.append(bt.runtime_ms)
        bt_found.append(bt.found)
        bt_costs.append(bt.cost if bt.found else float("nan"))

        mc = min_conflicts(csp, max_steps=10_000, max_restarts=5,
                           seed=seed, time_limit_ms=mc_timeout_ms)
        print(f"    MC: {mc.summary()}")
        mc_runtimes.append(mc.runtime_ms)
        mc_found.append(mc.found)
        mc_costs.append(mc.cost if mc.found else float("nan"))

    # main scaling plot — runtime vs N (this is the "money plot")
    plot_scaling(
        n_values=sizes,
        series_by_label={
            "Backtracking (MRV+LCV+FC)": bt_runtimes,
            "Min-Conflicts": mc_runtimes,
        },
        ylabel="Runtime (ms)",
        title="Run B — Runtime vs # users (BT vs Min-Conflicts)",
        output_path=str(out / "B_runtime_vs_N.png"),
        log_y=True,
    )

    # cost-when-found
    plot_scaling(
        n_values=sizes,
        series_by_label={
            "Backtracking": bt_costs,
            "Min-Conflicts": mc_costs,
        },
        ylabel="Total cost",
        title="Run B — Cost vs # users (where solver found something)",
        output_path=str(out / "B_cost_vs_N.png"),
        log_y=False,
    )

    # CSV
    with (out / "B_scaling.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_users", "bt_found", "bt_runtime_ms", "bt_cost",
                    "mc_found", "mc_runtime_ms", "mc_cost"])
        for i, n in enumerate(sizes):
            w.writerow([n, bt_found[i], f"{bt_runtimes[i]:.2f}",
                        f"{bt_costs[i]:.1f}" if bt_found[i] else "",
                        mc_found[i], f"{mc_runtimes[i]:.2f}",
                        f"{mc_costs[i]:.1f}" if mc_found[i] else ""])

    print(f"\n  -> {out}")
    return {"sizes": sizes, "bt_runtimes": bt_runtimes, "mc_runtimes": mc_runtimes,
            "bt_found": bt_found, "mc_found": mc_found, "out": str(out)}


# ---------------------------------------------------------------------------
# Run C — realistic, full Dhaka, Min-Conflicts only
# ---------------------------------------------------------------------------

def run_C(
    n_users: int = 150,
    max_stations: int = 30,
    seed: int = 42,
    time_slot: str = "10:00-12:00",
    mode: str = "normal",
) -> dict:
    out = OUTPUT_ROOT / "runC_full_dhaka"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[Run C] full Dhaka | n_users={n_users} | max_stations={max_stations} | mode={mode}")

    # policy presets
    policy_map = {
        "normal":    Policy(today_parity="any", min_service_quota=0),
        "rationing": Policy(today_parity="odd", min_service_quota=0),
        "strict":    Policy(today_parity="odd", min_service_quota=1, queue_cap=6,
                            strategic_reserve_fraction=0.15),
        "emergency": Policy(today_parity="any", min_service_quota=0,
                            public_reserve_fraction=0.30),
    }
    policy = policy_map.get(mode, policy_map["normal"])

    world = build_world(
        n_users=n_users, time_slot=time_slot, seed=seed,
        bbox=None,                       # full Dhaka
        max_stations=max_stations,
        area_name="full_dhaka",
    )
    csp = build_csp(world, policy=policy)

    mc = min_conflicts(csp, max_steps=20_000, max_restarts=5, seed=seed,
                       time_limit_ms=60_000)
    print(f"  MC: {mc.summary()}")

    plot_assignment_map(
        world, mc.assignment, mc.unassignable,
        title=f"Run C — full Dhaka | mode={mode} | cost={mc.cost:.0f} | "
              f"served={len(mc.assignment)}/{n_users}",
        output_path=str(out / f"C_map_{mode}.png"),
        bbox=None,
    )
    plot_station_loads(
        world, mc.assignment,
        policy_strategic_frac=policy.strategic_reserve_fraction,
        policy_public_frac=policy.public_reserve_fraction,
        output_path=str(out / f"C_loads_{mode}.png"),
    )
    plot_convergence(mc.convergence, output_path=str(out / f"C_convergence_{mode}.png"))

    with (out / f"C_summary_{mode}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mode", mode])
        w.writerow(["n_users", n_users])
        w.writerow(["max_stations", max_stations])
        w.writerow(["found", mc.found])
        w.writerow(["cost", f"{mc.cost:.1f}"])
        w.writerow(["served", len(mc.assignment)])
        w.writerow(["unassignable", len(mc.unassignable)])
        w.writerow(["iterations", mc.iterations])
        w.writerow(["runtime_ms", f"{mc.runtime_ms:.2f}"])

    print(f"  -> {out}")
    return {"mc": mc, "out": str(out)}


# ---------------------------------------------------------------------------
# Run DEMO — slide-aligned scoped comparison
#   * BT on a small portion of the network (Dhanmondi, ~12 users)
#   * Min-Conflicts on the WHOLE map (full Dhaka, ~150 users)
# This is the teacher's intended demo: BT proves correctness on a small
# instance; Min-Conflicts shows scalability on the city-wide instance
# where BT would never finish.
# ---------------------------------------------------------------------------

def run_demo(
    bt_users: int = 12,
    mc_users: int = 150,
    mc_max_stations: int = 30,
    seed: int = 42,
    time_slot: str = "10:00-12:00",
) -> dict:
    out = OUTPUT_ROOT / "run_demo_scoped"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[Run DEMO] BT on small portion (Dhanmondi, {bt_users} users) | "
          f"MC on whole map ({mc_users} users, full Dhaka)")

    permissive = Policy(today_parity="any", min_service_quota=0)

    # --- Method 1: BT on a small portion (Dhanmondi) ---
    print("\n[demo]  Method 1 (Backtracking + MRV + Degree + LCV + FC)  -- SMALL portion")
    bt_world = build_world(
        n_users=bt_users, time_slot=time_slot, seed=seed,
        bbox=DHANMONDI, max_stations=6, area_name="dhanmondi",
    )
    bt_csp = build_csp(bt_world, policy=permissive)
    bt = backtracking_search(bt_csp, use_mrv=True, use_lcv=True, use_fc=True,
                             time_limit_ms=30_000)
    print(f"  {bt.summary()}")
    plot_assignment_map(
        bt_world, bt.assignment, bt.unassignable,
        title=f"Method 1 (BT) — SMALL Dhanmondi portion | cost={bt.cost:.0f} | "
              f"served={len(bt.assignment)}/{bt_users}",
        output_path=str(out / "demo_bt_small.png"),
        bbox=DHANMONDI,
    )
    plot_station_loads(bt_world, bt.assignment, output_path=str(out / "demo_bt_small_loads.png"))

    # --- Method 2: Min-Conflicts on the WHOLE map ---
    print("\n[demo]  Method 2 (Min-Conflicts local search)  -- FULL Dhaka")
    mc_world = build_world(
        n_users=mc_users, time_slot=time_slot, seed=seed,
        bbox=None, max_stations=mc_max_stations, area_name="full_dhaka",
    )
    mc_csp = build_csp(mc_world, policy=permissive)
    mc = min_conflicts(mc_csp, max_steps=20_000, max_restarts=5,
                       seed=seed, time_limit_ms=60_000)
    print(f"  {mc.summary()}")
    plot_assignment_map(
        mc_world, mc.assignment, mc.unassignable,
        title=f"Method 2 (Min-Conflicts) — WHOLE Dhaka | cost={mc.cost:.0f} | "
              f"served={len(mc.assignment)}/{mc_users}",
        output_path=str(out / "demo_mc_large.png"),
        bbox=None,
    )
    plot_station_loads(mc_world, mc.assignment, output_path=str(out / "demo_mc_large_loads.png"))
    plot_convergence(mc.convergence, output_path=str(out / "demo_mc_convergence.png"))

    # --- Bonus: also show BT TRYING the large case so the scaling story is visible ---
    print("\n[demo]  (bonus) BT attempting the WHOLE-Dhaka case with a 15s timeout")
    bt_attempt = backtracking_search(mc_csp, use_mrv=True, use_lcv=True, use_fc=True,
                                     time_limit_ms=15_000)
    print(f"  {bt_attempt.summary()}")

    # --- summary CSV ---
    with (out / "demo_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "scope", "n_users", "found", "cost",
                    "served", "infeasible", "runtime_ms", "extra"])
        w.writerow(["Method 1 (BT + 3 heuristics)", "small (Dhanmondi)", bt_users,
                    bt.found, f"{bt.cost:.1f}", len(bt.assignment),
                    len(bt.unassignable), f"{bt.runtime_ms:.1f}",
                    f"expanded={bt.nodes_expanded} backtracks={bt.backtracks}"])
        w.writerow(["Method 2 (Min-Conflicts)", "whole map (full Dhaka)", mc_users,
                    mc.found, f"{mc.cost:.1f}", len(mc.assignment),
                    len(mc.unassignable), f"{mc.runtime_ms:.1f}",
                    f"iters={mc.iterations} violations={mc.final_violations}"])
        w.writerow(["Method 1 (BT) on whole map", "whole map (timeout demo)", mc_users,
                    bt_attempt.found, f"{bt_attempt.cost:.1f}" if bt_attempt.found else "",
                    len(bt_attempt.assignment), len(bt_attempt.unassignable),
                    f"{bt_attempt.runtime_ms:.1f}",
                    f"expanded={bt_attempt.nodes_expanded} backtracks={bt_attempt.backtracks} "
                    f"timed_out={bt_attempt.timed_out}"])

    print(f"\n[demo] Comparison written to {out}")
    print("\n--- Slide-aligned story ---")
    print(f"  BT on SMALL (Dhanmondi, {bt_users} users):      "
          f"found={bt.found}, cost={bt.cost:.0f}, runtime={bt.runtime_ms:.1f}ms")
    print(f"  MC on WHOLE (full Dhaka, {mc_users} users):     "
          f"found={mc.found}, cost={mc.cost:.0f}, runtime={mc.runtime_ms:.1f}ms")
    print(f"  BT attempted WHOLE (timeout = scaling proof):  "
          f"found={bt_attempt.found}, runtime={bt_attempt.runtime_ms:.1f}ms"
          f"{' (TIMED OUT)' if bt_attempt.timed_out else ''}")
    return {"bt_small": bt, "mc_large": mc, "bt_large_attempt": bt_attempt, "out": str(out)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["A", "B", "C", "demo", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-A", type=int, default=12)
    p.add_argument("--sizes-B", type=int, nargs="*", default=[10, 20, 40, 80, 160])
    p.add_argument("--bt-timeout-ms-B", type=float, default=30_000)
    p.add_argument("--mc-timeout-ms-B", type=float, default=30_000)
    p.add_argument("--n-C", type=int, default=150)
    p.add_argument("--stations-C", type=int, default=30)
    p.add_argument("--mode-C", default="normal",
                   choices=["normal", "rationing", "strict", "emergency"])
    p.add_argument("--bt-users-demo", type=int, default=12)
    p.add_argument("--mc-users-demo", type=int, default=150)
    args = p.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.run in ("A", "all"):
        run_A(n_users=args.n_A, seed=args.seed)
    if args.run in ("B", "all"):
        run_B(sizes=args.sizes_B, bt_timeout_ms=args.bt_timeout_ms_B,
              mc_timeout_ms=args.mc_timeout_ms_B, seed=args.seed)
    if args.run in ("C", "all"):
        run_C(n_users=args.n_C, max_stations=args.stations_C,
              seed=args.seed, mode=args.mode_C)
    if args.run in ("demo", "all"):
        run_demo(bt_users=args.bt_users_demo, mc_users=args.mc_users_demo,
                 seed=args.seed)

    print(f"\n[done] All requested experiments in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
