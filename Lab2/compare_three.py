"""Apples-to-apples: naive BT vs heuristic BT vs Min-Conflicts on the SAME CSP."""
from __future__ import annotations
import csv, math

from gas_world import build_world
from csp_model import build_csp, Policy
from backtracking import backtracking_search
from min_conflicts import min_conflicts

# Same Dhanmondi bbox used elsewhere in the project.
BBOX = (23.735, 23.770, 90.360, 90.395)
# Realistic-but-feasible policy: real queue_cap=8 creates genuine search pressure
# (forces users to spread across stations), while parity=any + quota=0 keeps a
# solution reachable so we compare *effort*, not just joint failure.
POLICY = Policy(today_parity="any", min_service_quota=0, queue_cap=8)
SEED = 42
TIME_SLOT = "10:00-12:00"
TIME_LIMIT_MS = 20_000

SCENARIOS = [8, 20, 40, 70, 100]

rows = []
for n in SCENARIOS:
    world = build_world(n_users=n, time_slot=TIME_SLOT, seed=SEED, bbox=BBOX, max_stations=15)
    # ONE csp per n; every solver gets an identical fresh copy via rebuild
    def fresh():
        return build_csp(world, policy=POLICY)

    naive = backtracking_search(fresh(), use_mrv=False, use_lcv=False, use_fc=False, time_limit_ms=TIME_LIMIT_MS)
    heur  = backtracking_search(fresh(), use_mrv=True,  use_lcv=True,  use_fc=True,  time_limit_ms=TIME_LIMIT_MS)
    mc    = min_conflicts(fresh(), max_steps=20_000, max_restarts=5, seed=SEED, time_limit_ms=TIME_LIMIT_MS)

    def fmt_cost(c):
        return "inf" if c == math.inf else f"{c:.0f}"

    rows.append({
        "n_users": n,
        "naive_found": naive.found, "naive_cost": fmt_cost(naive.cost),
        "naive_backtracks": naive.backtracks, "naive_ms": round(naive.runtime_ms, 2), "naive_timeout": naive.timed_out,
        "heur_found": heur.found, "heur_cost": fmt_cost(heur.cost),
        "heur_backtracks": heur.backtracks, "heur_ms": round(heur.runtime_ms, 2), "heur_timeout": heur.timed_out,
        "mc_found": mc.found, "mc_cost": fmt_cost(mc.cost),
        "mc_iters": mc.iterations, "mc_ms": round(mc.runtime_ms, 2), "mc_timeout": mc.timed_out,
    })
    print(f"n={n:3d} | naive {naive.summary()}")
    print(f"        | heur  {heur.summary()}")
    print(f"        | mc    {mc.summary()}")
    print()

out = "cache/experiments/three_way_comparison.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"[saved] {out}")
