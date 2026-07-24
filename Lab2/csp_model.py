

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
import statistics
from typing import Iterable

from gas_world import GasWorld, PEAK_SLOTS, Station, User


# ---------------------------------------------------------------------------
# constraint registry — name -> default enabled
# ---------------------------------------------------------------------------

ALL_CONSTRAINTS = (
    "U1_reachability",
    "U2_emergency_only",
    "G1_plate_parity",
    "G3_public_reserve",
    "G4_strategic_reserve",
    "G6_peak_hour_cng",
    "G8_queue_cap",
    "G9_min_service",
    "B1_household",
)


def default_enabled() -> dict[str, bool]:
    return {name: True for name in ALL_CONSTRAINTS}


@dataclass
class Policy:
    """Scenario-level knobs the teacher can flip in the CLI / Streamlit."""
    today_parity: str = "odd"            # "odd" | "even" | "any"
    public_reserve_fraction: float = 0.60
    strategic_reserve_fraction: float = 0.10
    queue_cap: int = 8
    min_service_quota: int = 1
    household_fraction: float = 0.20      # fraction of users paired into households
    households_per_group: int = 2         # 2 = pairs, 3 = triples




def generate_households(
    users: list[User],
    fraction: float,
    group_size: int,
    seed: int,
) -> dict[int, int]:
    """Assign each user a household_id. Returns {user_id: household_id}.

    Most users live alone (unique household). A fraction get grouped into
    shared households of `group_size` so B1 has work to do.
    """
    rng = random.Random(seed ^ 0xBEEF)
    household_id: dict[int, int] = {u.id: u.id for u in users}  # default: alone
    pool = [u.id for u in users]
    rng.shuffle(pool)

    n_grouped = int(round(fraction * len(users)))
    n_grouped -= n_grouped % group_size  # round down to multiple
    grouped = pool[:n_grouped]
    next_h_id = max(u.id for u in users) + 1
    for i in range(0, n_grouped, group_size):
        members = grouped[i:i + group_size]
        for m in members:
            household_id[m] = next_h_id
        next_h_id += 1
    return household_id


def household_pairs(household_id: dict[int, int]) -> list[tuple[int, int]]:
    """All (i, j) user pairs that share a household, i < j."""
    by_h: dict[int, list[int]] = {}
    for uid, hid in household_id.items():
        by_h.setdefault(hid, []).append(uid)
    pairs: list[tuple[int, int]] = []
    for members in by_h.values():
        if len(members) < 2:
            continue
        members.sort()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs



@dataclass
class CSP:
    world: GasWorld
    policy: Policy
    enabled: dict[str, bool]
    domains: dict[int, set[int]] = field(default_factory=dict)        # user_id -> set of station_ids
    household_id: dict[int, int] = field(default_factory=dict)
    _hh_neighbors: dict[int, list[int]] = field(default_factory=dict) # user_id -> household co-members

    @property
    def users(self) -> list[User]:
        return self.world.users

    @property
    def stations(self) -> list[Station]:
        return self.world.stations

    def user_by_id(self, uid: int) -> User:
        return self.world.users[uid]

    def station_by_id(self, sid: int) -> Station:
        return self.world.stations[sid]

    
    def build_initial_domains(self) -> None:
        peak = self.world.time_slot in PEAK_SLOTS
        for u in self.world.users:
            allowed: set[int] = set()
            for s in self.world.stations:
                if self.enabled["U1_reachability"] and not self.world.reachable(u, s):
                    continue
                if self.enabled["U2_emergency_only"] and s.is_emergency_only and u.vehicle_class != "emergency":
                    continue
                if self.enabled["G6_peak_hour_cng"] and peak and s.is_cng and u.vehicle_class != "emergency":
                    continue
                allowed.add(s.id)
            # G1 plate parity (whole-domain wipe for off-parity civilians)
            if (
                self.enabled["G1_plate_parity"]
                and self.policy.today_parity != "any"
                and u.plate_parity != self.policy.today_parity
                and u.vehicle_class != "emergency"
            ):
                allowed = set()
            self.domains[u.id] = allowed

    
    def _household_neighbors(self) -> dict[int, list[int]]:
        if self._hh_neighbors:
            return self._hh_neighbors
        nbrs: dict[int, list[int]] = {u.id: [] for u in self.world.users}
        for a, b in household_pairs(self.household_id):
            nbrs[a].append(b)
            nbrs[b].append(a)
        self._hh_neighbors = nbrs
        return nbrs

    def _revise(self, xi: int, xj: int) -> bool:
        """Standard AC-3 REVISE for the not-equal household constraint."""
        revised = False
        di = self.domains[xi]
        dj = self.domains[xj]
        # for each value x in di, if no value y in dj allows x != y, remove x
        to_remove = []
        for x in di:
            ok = any(y != x for y in dj)
            if not ok:
                to_remove.append(x)
        if to_remove:
            for x in to_remove:
                di.discard(x)
            revised = True
        return revised

    def ac3(self) -> bool:
        """Run AC-3 on B1 household pairs. Returns False on inconsistency."""
        if not self.enabled["B1_household"]:
            return True
        nbrs = self._household_neighbors()
        queue: deque[tuple[int, int]] = deque()
        for a, b in household_pairs(self.household_id):
            queue.append((a, b))
            queue.append((b, a))
        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj):
                if not self.domains[xi]:
                    return False
                for xk in nbrs[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

   

    def _station_state(self, assignment: dict[int, int]) -> dict[int, dict]:
        """For each station_id, accumulate: queue, total/private/public/emergency demand."""
        st: dict[int, dict] = {
            s.id: {"queue": 0, "total": 0.0, "private": 0.0, "public": 0.0, "emergency": 0.0}
            for s in self.world.stations
        }
        for uid, sid in assignment.items():
            if sid is None:
                continue
            u = self.world.users[uid]
            entry = st[sid]
            entry["queue"] += 1
            entry["total"] += u.demand_liters
            entry[u.vehicle_class] += u.demand_liters
        return st

    def _station_violates_with(
        self, station_state: dict, station: Station, new_user: User
    ) -> bool:
        """Would adding `new_user` to `station` violate any active global constraint?"""
        q = station_state["queue"] + 1
        tot = station_state["total"] + new_user.demand_liters
        priv = station_state["private"] + (new_user.demand_liters if new_user.vehicle_class == "private" else 0.0)
        pub = station_state["public"] + (new_user.demand_liters if new_user.vehicle_class == "public" else 0.0)

        if self.enabled["G8_queue_cap"] and q > self.policy.queue_cap:
            return True

        # capacity envelope
        cap = station.capacity_liters
        if self.enabled["G4_strategic_reserve"]:
            cap_civ = cap * (1.0 - self.policy.strategic_reserve_fraction)
        else:
            cap_civ = cap

        if new_user.vehicle_class == "emergency":
            # emergency may use full capacity (including reserve)
            if tot > cap:
                return True
        else:
            # civilians (private+public) must fit in cap_civ
            if (priv + pub) > cap_civ:
                return True
            if self.enabled["G3_public_reserve"]:
                # private may use at most (1 - public_reserve_fraction) of cap_civ
                priv_cap = cap_civ * (1.0 - self.policy.public_reserve_fraction)
                if priv > priv_cap:
                    return True
        return False

  

    def is_partial_consistent(
        self,
        assignment: dict[int, int],
        new_user: User,
        new_station_id: int,
    ) -> bool:
        """Can we extend `assignment` with new_user → new_station_id?

        Checks:
          * value in domain (caller usually ensures this)
          * B1 household (if active)
          * global capacity / queue (G3', G4, G8) at that station
        """
        if new_station_id not in self.domains[new_user.id]:
            return False

        # B1 — same household already at this station?
        if self.enabled["B1_household"]:
            for other in self._household_neighbors().get(new_user.id, []):
                if assignment.get(other) == new_station_id:
                    return False

        # global station capacity
        state = self._station_state(assignment)
        s_state = state[new_station_id]
        station = self.world.stations[new_station_id]
        if self._station_violates_with(s_state, station, new_user):
            return False
        return True

    def is_complete_consistent(self, assignment: dict[int, int]) -> bool:
        """All variables assigned + all constraints (including G9 min-service) satisfied."""
        if len(assignment) != len(self.world.users):
            return False
        if any(v is None for v in assignment.values()):
            return False

        # B1
        if self.enabled["B1_household"]:
            for a, b in household_pairs(self.household_id):
                if assignment.get(a) == assignment.get(b):
                    return False

        # G3'/G4/G8 — rebuild station state once
        state = self._station_state(assignment)
        for s in self.world.stations:
            entry = state[s.id]
            if self.enabled["G8_queue_cap"] and entry["queue"] > self.policy.queue_cap:
                return False
            cap = s.capacity_liters
            cap_civ = cap * (1.0 - self.policy.strategic_reserve_fraction) if self.enabled["G4_strategic_reserve"] else cap
            if (entry["private"] + entry["public"]) > cap_civ:
                return False
            if entry["total"] > cap:
                return False
            if self.enabled["G3_public_reserve"]:
                priv_cap = cap_civ * (1.0 - self.policy.public_reserve_fraction)
                if entry["private"] > priv_cap:
                    return False

        # G9 min-service — every station must serve >= quota
        if self.enabled["G9_min_service"]:
            for s in self.world.stations:
                if state[s.id]["queue"] < self.policy.min_service_quota:
                    return False

        return True

    # -----------------------------------------------------------------------
    # For Min-Conflicts: count violations
    # -----------------------------------------------------------------------

    def count_violations(self, assignment: dict[int, int]) -> int:
        """Total number of constraint violations under `assignment`.

        For unary constraints we count by "value not in domain".
        For B1 we count each conflicting pair once.
        For globals we count each over-capacity / over-queue station as 1 violation per excess unit.
        For G9 each station below quota counts as 1.
        """
        v = 0
        # value not in (pre-filtered) domain — captures unary violations baked in
        for uid, sid in assignment.items():
            if sid is None:
                v += 1
            elif sid not in self.domains[uid]:
                v += 1

        # B1
        if self.enabled["B1_household"]:
            for a, b in household_pairs(self.household_id):
                if assignment.get(a) is not None and assignment.get(a) == assignment.get(b):
                    v += 1

        state = self._station_state(assignment)
        for s in self.world.stations:
            entry = state[s.id]
            cap = s.capacity_liters
            cap_civ = cap * (1.0 - self.policy.strategic_reserve_fraction) if self.enabled["G4_strategic_reserve"] else cap
            if self.enabled["G8_queue_cap"] and entry["queue"] > self.policy.queue_cap:
                v += entry["queue"] - self.policy.queue_cap
            civ_demand = entry["private"] + entry["public"]
            if civ_demand > cap_civ:
                v += 1
            if entry["total"] > cap:
                v += 1
            if self.enabled["G3_public_reserve"]:
                priv_cap = cap_civ * (1.0 - self.policy.public_reserve_fraction)
                if entry["private"] > priv_cap:
                    v += 1
            if self.enabled["G9_min_service"] and entry["queue"] < self.policy.min_service_quota:
                v += 1
        return v

    def violations_if(
        self,
        assignment: dict[int, int],
        user: User,
        station_id: int,
    ) -> int:
        """Min-conflicts helper: count violations if `user` moves to `station_id`."""
        original = assignment.get(user.id)
        assignment[user.id] = station_id
        v = self.count_violations(assignment)
        if original is None:
            del assignment[user.id]
        else:
            assignment[user.id] = original
        return v

    # -----------------------------------------------------------------------
    # Soft cost
    # -----------------------------------------------------------------------

    def total_cost(
        self,
        assignment: dict[int, int],
        w_distance: float = 1.0,
        w_load: float = 500.0,  # load penalty scaled to roughly match meters
    ) -> float:
        """S1 distance + S2 load imbalance. Lower is better."""
        dist_sum = 0.0
        for uid, sid in assignment.items():
            if sid is None:
                continue
            dist_sum += self.world.distance(self.world.users[uid], self.world.stations[sid])

        # load variance across stations
        state = self._station_state(assignment)
        loads = [state[s.id]["total"] / max(1.0, s.capacity_liters) for s in self.world.stations]
        if len(loads) > 1:
            load_var = statistics.pvariance(loads)
        else:
            load_var = 0.0
        return w_distance * dist_sum + w_load * load_var


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_csp(
    world: GasWorld,
    policy: Policy | None = None,
    enabled: dict[str, bool] | None = None,
    household_seed: int | None = None,
) -> CSP:
    policy = policy or Policy()
    enabled = enabled or default_enabled()
    seed = household_seed if household_seed is not None else world.seed

    household_id = generate_households(
        world.users,
        fraction=policy.household_fraction,
        group_size=policy.households_per_group,
        seed=seed,
    )
    csp = CSP(
        world=world,
        policy=policy,
        enabled=enabled,
        household_id=household_id,
    )
    csp.build_initial_domains()
    csp.ac3()
    return csp


# ---------------------------------------------------------------------------
# Pretty-printer for the CLI
# ---------------------------------------------------------------------------

def print_domain_summary(csp: CSP) -> None:
    print(f"\n[csp] {len(csp.users)} users, {len(csp.stations)} stations")
    print(f"[csp] active constraints: {[k for k, v in csp.enabled.items() if v]}")
    hh = household_pairs(csp.household_id)
    print(f"[csp] {len(hh)} household pairs under B1")
    empty = sum(1 for uid, d in csp.domains.items() if not d)
    total = sum(len(d) for d in csp.domains.values())
    avg = total / max(1, len(csp.domains))
    print(f"[csp] domain sizes after unary + AC-3: empty={empty}, avg={avg:.2f}")
    sizes = sorted((len(d), uid) for uid, d in csp.domains.items())
    print(f"[csp] smallest-domain users (top 5): {[(uid, sz) for sz, uid in sizes[:5]]}")
    print(f"[csp] largest-domain users  (top 5): {[(uid, sz) for sz, uid in sizes[-5:]]}")


if __name__ == "__main__":
    from gas_world import build_world

    bbox = (23.735, 23.770, 90.360, 90.395)
    world = build_world(n_users=15, time_slot="10:00-12:00", seed=42, bbox=bbox, max_stations=8)

    # Two passes: "any" (rationing off) and "odd" (strict mode). The model is
    # correct in both; the strict mode just shows how aggressive policy can
    # wipe out feasibility — a useful demo for the report.
    for parity in ("any", "odd"):
        print(f"\n========== today_parity = {parity!r} ==========")
        policy = Policy(today_parity=parity)
        csp = build_csp(world, policy=policy)
        print_domain_summary(csp)
        feasible_users = sum(1 for d in csp.domains.values() if d)
        print(f"[csp] {feasible_users}/{len(csp.users)} users still have a non-empty domain")
        if parity == "any":
            print("\n[sample domains]")
            for u in csp.users[:8]:
                d = csp.domains[u.id]
                names = [csp.stations[sid].name[:20] for sid in sorted(d)]
                print(
                    f"  user {u.id:2d} | {u.vehicle_class:9s} | parity={u.plate_parity} | "
                    f"|D|={len(d)} | {names}"
                )
