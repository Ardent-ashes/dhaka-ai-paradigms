from dataclasses import dataclass
from typing import Any
import math
import random

NARROW_TYPES = {"residential", "living_street", "service", "track", "unclassified"}
PEAK_SLOTS = {"07:00-10:00", "17:00-21:00"}
MODERATE_SLOTS = {"10:00-12:00", "15:00-17:00"}

@dataclass
class Scenario:
    vehicle: str
    time_slot: str
    seed: int
    gender: str = "male"
    alone: bool = False
    pace: str = "relaxed"
    alpha: float = 1.0
    beta: float = 60.0
    gamma: float = 120.0
    delta_safety: float = 80.0
    accident_weight: float = 65.0
    weight_profile: str = "balanced"
    distance_priority: float = 3.0
    vehicle_priority: float = 2.0
    traffic_priority: float = 3.0
    safety_priority: float = 2.0
    accident_priority: float = 2.0


def _as_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)}


def _is_narrow(highway_value: Any) -> bool:
    return len(_as_set(highway_value).intersection(NARROW_TYPES)) > 0


def _base_traffic_level(time_slot: str) -> float:
    if time_slot in PEAK_SLOTS:
        return 0.9
    if time_slot in MODERATE_SLOTS:
        return 0.6
    return 0.3


def _accident_time_multiplier(time_slot: str) -> float:
    if time_slot in PEAK_SLOTS:
        return 1.35
    if time_slot in MODERATE_SLOTS:
        return 1.15
    return 1.0


def _profile_priorities(profile: str) -> tuple[float, float, float, float, float]:
    presets = {
        "balanced": (3.0, 2.0, 3.0, 2.0, 2.0),
        "fastest": (2.0, 1.0, 4.0, 1.0, 1.0),
        "safest": (2.0, 1.0, 2.0, 4.0, 4.0),
        "distance": (5.0, 1.0, 1.0, 1.0, 1.0),
    }
    return presets.get(profile, presets["balanced"])


def _normalized_pref_weights(scenario: Scenario) -> tuple[float, float, float, float, float]:
    if scenario.weight_profile == "custom":
        raw = [
            scenario.distance_priority,
            scenario.vehicle_priority,
            scenario.traffic_priority,
            scenario.safety_priority,
            scenario.accident_priority,
        ]
    else:
        raw = list(_profile_priorities(scenario.weight_profile))

    raw = [max(0.0, float(v)) for v in raw]
    s = sum(raw)
    if s <= 0:
        return 0.2, 0.2, 0.2, 0.2, 0.2
    return tuple(v / s for v in raw)


def _exp_utility(x: float, tau: float) -> float:
    """Smooth utility in [0,1): 1-exp(-x/tau)."""
    x = max(0.0, x)
    tau = max(1e-9, tau)
    return 1.0 - math.exp(-x / tau)


def road_intrinsic_risk_01(highway_value: Any) -> float:
    """Synthetic accident tendency proxy (mix of severity + conflict exposure)."""
    hset = _as_set(highway_value)
    if hset & {"motorway", "trunk"}:
        return 0.62
    if hset & {"primary"}:
        return 0.52
    if hset & {"secondary", "tertiary"}:
        return 0.44
    if hset & NARROW_TYPES:
        return 0.58
    if hset & {"residential", "unclassified"}:
        return 0.50
    return 0.45


def personal_safety_risk_01(highway_value: Any) -> float:
    """Synthetic personal-security proxy (visibility/publicness vs narrow/isolated)."""
    hset = _as_set(highway_value)
    if hset & {"motorway", "trunk", "primary"}:
        return 0.18
    if hset & {"secondary", "tertiary"}:
        return 0.30
    if hset & {"residential", "unclassified"}:
        return 0.55
    if hset & NARROW_TYPES:
        return 0.78
    return 0.42


def _user_safety_sensitivity(scenario: Scenario, narrow: bool) -> float:
    """Higher = care more about risky edges (used only in edge_cost, not heuristic)."""
    g = scenario.gender.lower()
    s = 0.0
    if g == "female" and scenario.alone:
        s = 1.0
    elif g == "female":
        s = 0.45
    elif scenario.alone:
        s = 0.35
    if narrow:
        s = min(1.0, s + 0.15)
    if scenario.pace == "rush":
        s *= 0.55
    return s


def edge_cost(edge_data: dict, scenario: Scenario) -> float:
    length = float(edge_data.get("length", 1.0))
    highway = edge_data.get("highway", "residential")
    narrow = _is_narrow(highway)
    accident_risk01 = road_intrinsic_risk_01(highway)
    personal_risk01 = personal_safety_risk_01(highway)

    if scenario.vehicle == "bus" and narrow:
        return math.inf

    vehicle_penalty = 1.0
    if scenario.vehicle == "bus" and "primary" not in _as_set(highway):
        vehicle_penalty = 1.4
    elif scenario.vehicle == "car" and narrow:
        vehicle_penalty = 2.2

    rnd = random.Random(scenario.seed + int(length) % 97)
    jam_noise = rnd.uniform(0.0, 0.5)
    traffic_level = min(1.5, _base_traffic_level(scenario.time_slot) + jam_noise)
    if scenario.pace == "rush":
        traffic_level = min(1.5, traffic_level + 0.12)

    sens = _user_safety_sensitivity(scenario, narrow)
    safety_level = personal_risk01 * sens


    accident_noise = rnd.uniform(0.0, 0.2)
    accident_level = min(
        1.8,
        (accident_risk01 * _accident_time_multiplier(scenario.time_slot)) + accident_noise,
    )
    dist_u = _exp_utility(length, 300.0)
    vehicle_u = _exp_utility(max(0.0, vehicle_penalty - 1.0), 0.6)
    traffic_u = _exp_utility(traffic_level, 0.8)
    safety_u = _exp_utility(safety_level, 0.35)
    accident_u = _exp_utility(accident_level, 1.0)

    wd, wv, wt, ws, wa = _normalized_pref_weights(scenario)
    combined_norm = (
        (wd * dist_u)
        + (wv * vehicle_u)
        + (wt * traffic_u)
        + (ws * safety_u)
        + (wa * accident_u)
    )
    scale = (
        scenario.alpha * 300.0
        + scenario.beta * 2.2
        + scenario.gamma * 1.5
        + scenario.delta_safety
        + scenario.accident_weight * 1.8
    )
    return combined_norm * scale


def edge_cost_lower_bound(edge_data: dict, scenario: Scenario) -> float:
    """Deterministic lower bound per edge for admissible A* heuristic.

    This intentionally drops non-negative random terms and user-specific safety terms.
    """
    length = float(edge_data.get("length", 1.0))
    highway = edge_data.get("highway", "residential")
    accident_risk01 = road_intrinsic_risk_01(highway)
    traffic_level_lb = _base_traffic_level(scenario.time_slot)
    accident_level_lb = min(
        1.8, accident_risk01 * _accident_time_multiplier(scenario.time_slot)
    )

    dist_u_lb = _exp_utility(length, 300.0)
    vehicle_u_lb = _exp_utility(0.0, 0.6)
    traffic_u_lb = _exp_utility(traffic_level_lb, 0.8)
    safety_u_lb = _exp_utility(0.0, 0.35)
    accident_u_lb = _exp_utility(accident_level_lb, 1.0)

    wd, wv, wt, ws, wa = _normalized_pref_weights(scenario)
    combined_norm_lb = (
        (wd * dist_u_lb)
        + (wv * vehicle_u_lb)
        + (wt * traffic_u_lb)
        + (ws * safety_u_lb)
        + (wa * accident_u_lb)
    )
    scale = (
        scenario.alpha * 300.0
        + scenario.beta * 2.2
        + scenario.gamma * 1.5
        + scenario.delta_safety
        + scenario.accident_weight * 1.8
    )
    return combined_norm_lb * scale


def heuristic_edge_cost(edge_data: dict, scenario: Scenario) -> float:
    """Heuristic edge weight used for reverse-Dijkstra cache.

    Kept as edge-wise lower bound of `edge_cost` so that h(n) is admissible.
    """
    return edge_cost_lower_bound(edge_data, scenario)


def heuristic_distance(graph, current, goal) -> float:
    y1, x1 = graph.nodes[current]["y"], graph.nodes[current]["x"]
    y2, x2 = graph.nodes[goal]["y"], graph.nodes[goal]["x"]
    return ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5 * 111_000.0
    


def build_euclidean_practical_cache(graph, goal, scenario: Scenario) -> dict:
    """Euclidean heuristic with local road-wise signals.

    Uses Euclidean distance plus:
    - base traffic by time slot
    - local road risk proxy from incident edges' OSM `highway` tags
    """

    def _incident_edge_risk(node) -> float:
        risks = []
        total = 0
        try:
            it = graph.out_edges(node, data=True) if graph.is_directed() else graph.edges(node, data=True)
        except Exception:
            it = []
        for _, _, data in it:
            highway = data.get("highway", "residential")
            risks.append(road_intrinsic_risk_01(highway))
            total += 1
            if total >= 12:
                break
        if total == 0:
            return 0.45
        return sum(risks) / total

    cache: dict = {}
    base_t = _base_traffic_level(scenario.time_slot)
    wd, wv, wt, ws, wa = _normalized_pref_weights(scenario)
    total_w = wd + wt + wa
    if total_w <= 0:
        wd_n, wt_n, wa_n = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    else:
        wd_n, wt_n, wa_n = wd / total_w, wt / total_w, wa / total_w

    for node in graph.nodes:
        d = heuristic_distance(graph, node, goal)
        local_risk = _incident_edge_risk(node)
        traffic_est = base_t
        accident_est = min(1.8, local_risk * _accident_time_multiplier(scenario.time_slot))
        dist_u = _exp_utility(d, 300.0)
        traffic_u = _exp_utility(traffic_est, 0.8)
        accident_u = _exp_utility(accident_est, 1.0)

        combined = (wd_n * dist_u) + (wt_n * traffic_u) + (wa_n * accident_u)
        scale = (
            (scenario.alpha * 300.0)
            + (scenario.gamma * 1.5)
            + (scenario.accident_weight * 1.8)
        )

        cache[node] = combined * scale
    return cache
