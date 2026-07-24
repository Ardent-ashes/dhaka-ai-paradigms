"""
min_conflicts.py — local search for CSP (slide deck's 3rd heuristic)
====================================================================

The classic Min-Conflicts algorithm:

  1. Start from a random *complete* assignment (every feasible user gets some
     station from their domain).
  2. Loop:
       * If no conflicts, return.
       * Pick a conflicted user (random among those involved in violations).
       * Reassign that user to the value (station) that minimizes total
         constraint violations.
       * Track best-so-far for restart fallback.
  3. On a plateau (no improvement for `patience` steps), random restart.

This scales to large instances where backtracking can't follow — exactly
the Run C / scaling-sweep story.

Returns an MCResult with the convergence curve so visualize.py can plot it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from time import perf_counter

from csp_model import CSP, household_pairs


@dataclass
class MCResult:
    found: bool
    assignment: dict[int, int] = field(default_factory=dict)
    unassignable: list[int] = field(default_factory=list)
    cost: float = math.inf
    iterations: int = 0
    final_violations: int = 0
    convergence: list[int] = field(default_factory=list)
    restarts: int = 0
    runtime_ms: float = 0.0
    timed_out: bool = False

    def summary(self) -> str:
        return (
            f"MinConflicts found={self.found} cost={self.cost:.1f} "
            f"iters={self.iterations} violations={self.final_violations} "
            f"restarts={self.restarts} runtime={self.runtime_ms:.1f}ms"
            + (" [TIMEOUT]" if self.timed_out else "")
        )


def _conflicted_users(csp: CSP, assignment: dict[int, int]) -> list[int]:
    """User ids involved in at least one constraint violation."""
    conflicted: set[int] = set()

    # B1 — same-household pairs sharing a station
    if csp.enabled["B1_household"]:
        for a, b in household_pairs(csp.household_id):
            if assignment.get(a) is not None and assignment.get(a) == assignment.get(b):
                conflicted.add(a)
                conflicted.add(b)

    # Global station overflows — any user at an offending station is conflicted
    state = csp._station_state(assignment)
    over_stations: set[int] = set()
    for s in csp.stations:
        entry = state[s.id]
        cap = s.capacity_liters
        cap_civ = (
            cap * (1.0 - csp.policy.strategic_reserve_fraction)
            if csp.enabled["G4_strategic_reserve"]
            else cap
        )
        if csp.enabled["G8_queue_cap"] and entry["queue"] > csp.policy.queue_cap:
            over_stations.add(s.id)
        if (entry["private"] + entry["public"]) > cap_civ:
            over_stations.add(s.id)
        if entry["total"] > cap:
            over_stations.add(s.id)
        if csp.enabled["G3_public_reserve"]:
            priv_cap = cap_civ * (1.0 - csp.policy.public_reserve_fraction)
            if entry["private"] > priv_cap:
                over_stations.add(s.id)
        if csp.enabled["G9_min_service"] and entry["queue"] < csp.policy.min_service_quota:
            # any user at *other* stations could move here — they're all "conflicted"
            # from G9's perspective. We approximate by tagging an arbitrary handful.
            pass  # handled implicitly by allowing reassignment to underused stations

    for uid, sid in assignment.items():
        if sid in over_stations:
            conflicted.add(uid)

    return list(conflicted)


def min_conflicts(
    csp: CSP,
    max_steps: int = 5_000,
    max_restarts: int = 5,
    patience: int = 500,
    seed: int = 42,
    time_limit_ms: float | None = None,
) -> MCResult:
    rng = random.Random(seed)
    t0 = perf_counter()
    deadline = (t0 + time_limit_ms / 1000.0) if time_limit_ms else None

    feasible = [u for u in csp.users if csp.domains[u.id]]
    unassignable = [u.id for u in csp.users if not csp.domains[u.id]]

    if not feasible:
        return MCResult(
            found=False, assignment={}, unassignable=unassignable,
            cost=math.inf, runtime_ms=(perf_counter() - t0) * 1000.0,
        )

    best_assignment: dict[int, int] = {}
    best_violations = math.inf
    convergence: list[int] = []
    total_iters = 0
    timed_out = False

    for restart in range(max_restarts + 1):
        # random initial assignment over feasible users
        assignment: dict[int, int] = {
            u.id: rng.choice(tuple(csp.domains[u.id])) for u in feasible
        }
        plateau = 0
        last_v = math.inf

        for step in range(max_steps):
            if deadline and perf_counter() > deadline:
                timed_out = True
                break

            total_iters += 1
            v = csp.count_violations(assignment)
            convergence.append(v)

            if v < best_violations:
                best_violations = v
                best_assignment = dict(assignment)
                plateau = 0
            else:
                plateau += 1

            if v == 0:
                runtime_ms = (perf_counter() - t0) * 1000.0
                return MCResult(
                    found=True,
                    assignment=dict(assignment),
                    unassignable=unassignable,
                    cost=csp.total_cost(assignment),
                    iterations=total_iters,
                    final_violations=0,
                    convergence=convergence,
                    restarts=restart,
                    runtime_ms=runtime_ms,
                )

            if plateau >= patience:
                break  # this restart has stalled

            # pick a conflicted user — if none flagged, perturb a random one
            conflicted = _conflicted_users(csp, assignment)
            if not conflicted:
                conflicted = [u.id for u in feasible]
            uid = rng.choice(conflicted)
            u = csp.user_by_id(uid)

            # min-conflicts value choice (ties broken randomly)
            candidates = list(csp.domains[uid])
            if not candidates:
                continue
            best_vio = math.inf
            best_values: list[int] = []
            for sid in candidates:
                vio = csp.violations_if(assignment, u, sid)
                if vio < best_vio:
                    best_vio = vio
                    best_values = [sid]
                elif vio == best_vio:
                    best_values.append(sid)
            assignment[uid] = rng.choice(best_values)
            last_v = v

        if timed_out:
            break

    runtime_ms = (perf_counter() - t0) * 1000.0
    found = best_violations == 0
    return MCResult(
        found=found,
        assignment=best_assignment,
        unassignable=unassignable,
        cost=csp.total_cost(best_assignment) if best_assignment else math.inf,
        iterations=total_iters,
        final_violations=int(best_violations) if math.isfinite(best_violations) else -1,
        convergence=convergence,
        restarts=max_restarts,
        runtime_ms=runtime_ms,
        timed_out=timed_out,
    )


if __name__ == "__main__":
    from gas_world import build_world
    from csp_model import build_csp, Policy, print_domain_summary

    bbox = (23.735, 23.770, 90.360, 90.395)
    world = build_world(n_users=15, time_slot="10:00-12:00", seed=42, bbox=bbox, max_stations=8)

    policy = Policy(today_parity="any", min_service_quota=0)
    csp = build_csp(world, policy=policy)
    print_domain_summary(csp)

    print("\n[mc] Running Min-Conflicts...")
    result = min_conflicts(csp, max_steps=2000, max_restarts=3, seed=42)
    print(f"  {result.summary()}")

    print(f"\n[mc] Convergence (first 30 steps): {result.convergence[:30]}")
    print(f"[mc] Final assignment ({len(result.assignment)} served, "
          f"{len(result.unassignable)} infeasible):")
    for uid in sorted(result.assignment):
        sid = result.assignment[uid]
        u = world.users[uid]
        s = world.stations[sid]
        d = world.distance(u, s)
        print(f"  user {uid:2d} ({u.vehicle_class:9s}) -> station {sid} ({s.name[:25]:25s}) | {d:7.0f}m")
