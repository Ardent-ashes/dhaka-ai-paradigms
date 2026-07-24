

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter

from csp_model import CSP


@dataclass
class BTResult:
    found: bool
    assignment: dict[int, int] = field(default_factory=dict)  # user_id -> station_id
    unassignable: list[int] = field(default_factory=list)     # user_ids with ∅ domain pre-search
    cost: float = math.inf
    nodes_expanded: int = 0
    backtracks: int = 0
    runtime_ms: float = 0.0
    timed_out: bool = False
    use_mrv: bool = True
    use_lcv: bool = True
    use_fc: bool = True

    def summary(self) -> str:
        # All 3 slide heuristics live inside this BT call:
        #   MRV (slide H1) = use_mrv -- pick variable with smallest domain
        #   Degree (slide H2) = applied as MRV tiebreaker on household neighbors
        #   LCV (slide H3) = use_lcv -- order values by least constraining
        # Forward Checking is the slide's "interleaving inference" step.
        heur = []
        if self.use_mrv:
            heur.append("MRV+Degree")
        if self.use_lcv:
            heur.append("LCV")
        if self.use_fc:
            heur.append("FC")
        tag = "+".join(heur) if heur else "plain"
        return (
            f"BT[{tag}] found={self.found} cost={self.cost:.1f} "
            f"expanded={self.nodes_expanded} backtracks={self.backtracks} "
            f"unassignable={len(self.unassignable)} "
            f"runtime={self.runtime_ms:.1f}ms"
            + (" [TIMEOUT]" if self.timed_out else "")
        )


class _Timeout(Exception):
    pass


def _select_unassigned(csp: CSP, assignment: dict[int, int], pending: set[int], use_mrv: bool) -> int | None:
    """Pick next user_id. MRV with degree-heuristic tiebreaker."""
    if not pending:
        return None
    if not use_mrv:
        # leftmost unassigned
        return min(pending)

    nbrs = csp._household_neighbors() if csp.enabled["B1_household"] else {}
    best_uid = None
    best_key = None
    for uid in pending:
        d_size = len(csp.domains[uid])
        deg = sum(1 for n in nbrs.get(uid, []) if n in pending)
        # smaller domain wins; ties broken by higher degree (more unassigned neighbors)
        key = (d_size, -deg, uid)
        if best_key is None or key < best_key:
            best_key = key
            best_uid = uid
    return best_uid


def _order_values(
    csp: CSP,
    assignment: dict[int, int],
    user_id: int,
    pending: set[int],
    use_lcv: bool,
) -> list[int]:
    """Order station ids for this user. LCV = try least-constraining station first."""
    candidates = list(csp.domains[user_id])
    if not use_lcv or not candidates:
        return candidates

    # household neighbors that would lose this value
    nbrs = csp._household_neighbors().get(user_id, []) if csp.enabled["B1_household"] else []
    u = csp.user_by_id(user_id)

    # Compute station state ONCE up front, not per-candidate.
    # Without this, LCV recomputes O(N) state for every station in the sort key,
    # which makes "heuristic BT" slower than "naive BT" on tight scenarios.
    state_snapshot = csp._station_state(assignment)
    strategic_frac = csp.policy.strategic_reserve_fraction

    def constrain_score(sid: int) -> tuple[int, float]:
        bin_removed = sum(1 for n in nbrs if n in pending and sid in csp.domains[n])
        s = csp.station_by_id(sid)
        cap_civ = s.capacity_liters * (1.0 - strategic_frac)
        st = state_snapshot[sid]
        headroom = cap_civ - (st["private"] + st["public"])
        return (bin_removed, -headroom)

    candidates.sort(key=constrain_score)
    return candidates


def _forward_check(
    csp: CSP,
    user_id: int,
    chosen_sid: int,
    pending: set[int],
) -> tuple[bool, dict[int, set[int]]]:
    """After assigning user_id <- chosen_sid, prune neighbor domains.

    Returns (still_consistent, removed_by_user) so we can undo on backtrack.
    """
    removed: dict[int, set[int]] = {}
    if not csp.enabled["B1_household"]:
        return True, removed
    nbrs = csp._household_neighbors().get(user_id, [])
    for n in nbrs:
        if n not in pending:
            continue
        if chosen_sid in csp.domains[n]:
            csp.domains[n].discard(chosen_sid)
            removed.setdefault(n, set()).add(chosen_sid)
            if not csp.domains[n]:
                return False, removed
    return True, removed


def _undo_forward_check(csp: CSP, removed: dict[int, set[int]]) -> None:
    for uid, vals in removed.items():
        csp.domains[uid] |= vals


def backtracking_search(
    csp: CSP,
    use_mrv: bool = True,
    use_lcv: bool = True,
    use_fc: bool = True,
    time_limit_ms: float | None = None,
) -> BTResult:
    t0 = perf_counter()
    deadline = (t0 + time_limit_ms / 1000.0) if time_limit_ms else None

    # snapshot pre-search domains so FC mutations don't bleed across runs
    snapshot = {uid: set(d) for uid, d in csp.domains.items()}

    unassignable = [u.id for u in csp.users if not csp.domains[u.id]]
    pending: set[int] = {u.id for u in csp.users if csp.domains[u.id]}
    assignment: dict[int, int] = {}
    stats = {"expanded": 0, "backtracks": 0}

    def recurse() -> bool:
        if deadline is not None and perf_counter() > deadline:
            raise _Timeout()
        stats["expanded"] += 1

        if not pending:
            # all reachable users assigned; check the global G9 / completeness check.
            # Treat unassignable as "not served" — G9 sees only assigned counts.
            return _final_check(csp, assignment, unassignable)

        uid = _select_unassigned(csp, assignment, pending, use_mrv)
        if uid is None:
            return False
        u = csp.user_by_id(uid)
        values = _order_values(csp, assignment, uid, pending, use_lcv)

        pending.discard(uid)
        for sid in values:
            if not csp.is_partial_consistent(assignment, u, sid):
                continue
            assignment[uid] = sid

            ok, removed = (True, {})
            if use_fc:
                ok, removed = _forward_check(csp, uid, sid, pending)

            if ok and recurse():
                return True

            if use_fc:
                _undo_forward_check(csp, removed)
            del assignment[uid]

        pending.add(uid)
        stats["backtracks"] += 1
        return False

    found = False
    timed_out = False
    try:
        found = recurse()
    except _Timeout:
        timed_out = True

    runtime_ms = (perf_counter() - t0) * 1000.0

    # restore domains
    for uid, d in snapshot.items():
        csp.domains[uid] = d

    cost = csp.total_cost(assignment) if found else math.inf
    return BTResult(
        found=found,
        assignment=dict(assignment) if found else {},
        unassignable=unassignable,
        cost=cost,
        nodes_expanded=stats["expanded"],
        backtracks=stats["backtracks"],
        runtime_ms=runtime_ms,
        timed_out=timed_out,
        use_mrv=use_mrv,
        use_lcv=use_lcv,
        use_fc=use_fc,
    )


def _final_check(csp: CSP, assignment: dict[int, int], unassignable: list[int]) -> bool:
    """Same as csp.is_complete_consistent but only over the feasible subset.

    Unassignable users are excluded (they're reported separately, not failed).
    """
    if csp.enabled["B1_household"]:
        from csp_model import household_pairs
        for a, b in household_pairs(csp.household_id):
            if a in assignment and b in assignment and assignment[a] == assignment[b]:
                return False

    state = csp._station_state(assignment)
    for s in csp.stations:
        entry = state[s.id]
        if csp.enabled["G8_queue_cap"] and entry["queue"] > csp.policy.queue_cap:
            return False
        cap = s.capacity_liters
        cap_civ = cap * (1.0 - csp.policy.strategic_reserve_fraction) if csp.enabled["G4_strategic_reserve"] else cap
        if (entry["private"] + entry["public"]) > cap_civ:
            return False
        if entry["total"] > cap:
            return False
        if csp.enabled["G3_public_reserve"]:
            priv_cap = cap_civ * (1.0 - csp.policy.public_reserve_fraction)
            if entry["private"] > priv_cap:
                return False
    if csp.enabled["G9_min_service"]:
        for s in csp.stations:
            if state[s.id]["queue"] < csp.policy.min_service_quota:
                return False
    return True


if __name__ == "__main__":
    from gas_world import build_world
    from csp_model import build_csp, Policy, print_domain_summary

    bbox = (23.735, 23.770, 90.360, 90.395)
    world = build_world(n_users=15, time_slot="10:00-12:00", seed=42, bbox=bbox, max_stations=8)

    policy = Policy(today_parity="any", min_service_quota=0)  # generous demo
    csp = build_csp(world, policy=policy)
    print_domain_summary(csp)

    print("\n[bt] Running 4 backtracking variants on the same CSP...")
    for use_mrv, use_lcv, use_fc in [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]:
        result = backtracking_search(csp, use_mrv=use_mrv, use_lcv=use_lcv, use_fc=use_fc, time_limit_ms=10_000)
        print(f"  {result.summary()}")

    # Show the best assignment
    result = backtracking_search(csp, use_mrv=True, use_lcv=True, use_fc=True)
    print(f"\n[bt] Final assignment ({len(result.assignment)}/{len(world.users)} served, "
          f"{len(result.unassignable)} infeasible):")
    for uid in sorted(result.assignment):
        sid = result.assignment[uid]
        u = world.users[uid]
        s = world.stations[sid]
        d = world.distance(u, s)
        print(f"  user {uid:2d} ({u.vehicle_class:9s}) -> station {sid} ({s.name[:25]:25s}) | {d:7.0f}m")
    if result.unassignable:
        print(f"  [unassignable] users {result.unassignable} — empty domain after AC-3")
