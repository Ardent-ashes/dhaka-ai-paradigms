

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import heapq
import json
import math
import random
from typing import Any

import osmnx as ox

from data_loader import load_dhaka_graph


# ---------------------------------------------------------------------------
# constants (some borrowed from Lab 1 cost_engine.py to stay consistent)
# ---------------------------------------------------------------------------

VEHICLE_CLASSES = ("private", "public", "emergency")
PLATE_PARITIES = ("odd", "even")

NARROW_TYPES = {"residential", "living_street", "service", "track", "unclassified"}
PEAK_SLOTS = {"07:00-10:00", "17:00-21:00"}
MODERATE_SLOTS = {"10:00-12:00", "15:00-17:00"}

# When a specific bbox preset is selected, every user/station in that bbox
# gets this short zone code as its `zone` label. For full-Dhaka mode the
# value is None and we fall back to the 3x3 lat/lon grid (Z00..Z22).
AREA_ZONE_CODE: dict[str, str | None] = {
    "dhanmondi":  "DH",
    "mohakhali":  "MK",
    "mirpur":     "MP",
    "old_dhaka":  "OD",
    "full_dhaka": None,
}


DEFAULT_HORIZON_METERS = 15_000.0


def _base_traffic_level(time_slot: str) -> float:
    if time_slot in PEAK_SLOTS:
        return 0.9
    if time_slot in MODERATE_SLOTS:
        return 0.6
    return 0.3


def _is_narrow(highway_value: Any) -> bool:
    if isinstance(highway_value, list):
        tags = {str(v) for v in highway_value}
    else:
        tags = {str(highway_value)}
    return len(tags & NARROW_TYPES) > 0




@dataclass
class Station:
    id: int
    node: int
    name: str
    lat: float
    lon: float
    zone: str
    capacity_liters: float
    is_cng: bool
    is_emergency_only: bool


@dataclass
class User:
    id: int
    node: int
    lat: float
    lon: float
    vehicle_class: str  # "private" | "public" | "emergency"
    zone: str
    plate_parity: str   # "odd" | "even"
    fuel_left_km: float
    demand_liters: float



def _make_zone_grid(graph, n_rows: int = 3, n_cols: int = 3):
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    bbox = (min(ys), max(ys), min(xs), max(xs))
    labels = [[f"Z{r}{c}" for c in range(n_cols)] for r in range(n_rows)]
    return bbox, labels


def _zone_of(lat: float, lon: float, bbox, labels) -> str:
    lat_min, lat_max, lon_min, lon_max = bbox
    n_rows, n_cols = len(labels), len(labels[0])
    span_lat = max(1e-9, lat_max - lat_min)
    span_lon = max(1e-9, lon_max - lon_min)
    r = min(n_rows - 1, max(0, int((lat - lat_min) / span_lat * n_rows)))
    c = min(n_cols - 1, max(0, int((lon - lon_min) / span_lon * n_cols)))
    return labels[r][c]



def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _fetch_osm_fuel_stations() -> list[dict]:
    print("[gas_world] Querying OSM for Dhaka fuel stations...")
    try:
        gdf = ox.features_from_place("Dhaka, Bangladesh", tags={"amenity": "fuel"})
    except AttributeError:
        # older osmnx <1.5
        gdf = ox.geometries_from_place("Dhaka, Bangladesh", tags={"amenity": "fuel"})

    gdf = gdf.copy()
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x

    rows: list[dict] = []
    for i, (_, row) in enumerate(gdf.iterrows()):
        name_val = row.get("name") if "name" in gdf.columns else None
        name = str(name_val) if name_val is not None and not _is_nan(name_val) else f"Station-{i}"
        is_cng = False
        if "fuel:cng" in gdf.columns:
            v = row.get("fuel:cng")
            is_cng = bool(str(v).lower() == "yes")
        rows.append({
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "name": name,
            "is_cng": is_cng,
        })
    return rows


def load_stations_from_osm(
    graph,
    cache_path: str = "cache/dhaka_fuel.json",
    bbox: tuple[float, float, float, float] | None = None,
    max_stations: int | None = None,
    seed: int = 42,
    area_zone_code: str | None = None,
) -> list[Station]:

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        raw = json.loads(cache_file.read_text())
        print(f"[gas_world] Loaded {len(raw)} cached fuel stations from {cache_file}")
    else:
        raw = _fetch_osm_fuel_stations()
        cache_file.write_text(json.dumps(raw, indent=2))
        print(f"[gas_world] Saved {len(raw)} stations to {cache_file}")

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        raw = [r for r in raw if lat_min <= r["lat"] <= lat_max and lon_min <= r["lon"] <= lon_max]
        print(f"[gas_world] After bbox filter: {len(raw)} stations")

    if max_stations is not None and len(raw) > max_stations:
        rng = random.Random(seed ^ 0xDEADBEEF)
        raw = rng.sample(raw, max_stations)
        print(f"[gas_world] Subsampled to {len(raw)} stations")

    zone_bbox, labels = _make_zone_grid(graph)
    stations: list[Station] = []
    for i, r in enumerate(raw):
        node = ox.distance.nearest_nodes(graph, X=r["lon"], Y=r["lat"])
        zone = area_zone_code if area_zone_code else _zone_of(r["lat"], r["lon"], zone_bbox, labels)
        stations.append(Station(
            id=i,
            node=node,
            name=r["name"],
            lat=r["lat"],
            lon=r["lon"],
            zone=zone,
            capacity_liters=0.0,
            is_cng=r["is_cng"],
            is_emergency_only=False,
        ))
    return stations


def configure_stations(
    stations: list[Station],
    seed: int,
    base_capacity_liters: float = 2000.0,
    capacity_jitter: float = 0.4,
    fraction_emergency_only: float = 0.08,
    fraction_cng_if_unknown: float = 0.35,
) -> None:
    """Fill in capacity + flags. OSM doesn't tell us capacity, so we synthesize."""
    rng = random.Random(seed ^ 0xC0FFEE)
    n = len(stations)
    n_emergency = max(1, int(round(fraction_emergency_only * n)))
    emergency_ids = set(rng.sample(range(n), n_emergency))
    for s in stations:
        s.capacity_liters = base_capacity_liters * (
            1.0 + rng.uniform(-capacity_jitter, capacity_jitter)
        )
        s.is_emergency_only = s.id in emergency_ids
        if not s.is_cng and rng.random() < fraction_cng_if_unknown:
            s.is_cng = True


def generate_users(
    graph,
    n_users: int,
    seed: int,
    bbox: tuple[float, float, float, float] | None = None,
    class_fractions: dict[str, float] | None = None,
    area_zone_code: str | None = None,
) -> list[User]:
  
    rng = random.Random(seed)
    nodes = list(graph.nodes)
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        nodes = [
            n for n in nodes
            if lat_min <= graph.nodes[n]["y"] <= lat_max
            and lon_min <= graph.nodes[n]["x"] <= lon_max
        ]
        if not nodes:
            raise ValueError("bbox produced empty node set — widen it")

    class_fractions = class_fractions or {"private": 0.60, "public": 0.30, "emergency": 0.10}
    vc_names = list(class_fractions.keys())
    vc_weights = list(class_fractions.values())

    zone_bbox, zone_labels = _make_zone_grid(graph)
    users: list[User] = []
    for i in range(n_users):
        node = rng.choice(nodes)
        lat = graph.nodes[node]["y"]
        lon = graph.nodes[node]["x"]
        zone = area_zone_code if area_zone_code else _zone_of(lat, lon, zone_bbox, zone_labels)
        users.append(User(
            id=i,
            node=node,
            lat=lat,
            lon=lon,
            vehicle_class=rng.choices(vc_names, weights=vc_weights, k=1)[0],
            zone=zone,
            plate_parity=rng.choice(PLATE_PARITIES),
            fuel_left_km=rng.uniform(3.0, 12.0),
            demand_liters=rng.uniform(10.0, 30.0),
        ))
    return users




def _min_edge(graph, u, v):
    data = graph.get_edge_data(u, v)
    if not data:
        return None
    if "length" in data:
        return data
    best_key = min(data.keys(), key=lambda k: float(data[k].get("length", 1.0)))
    return data[best_key]


def _edge_weight(edge_data: dict, vehicle_class: str, traffic_level: float) -> float:
    length = float(edge_data.get("length", 1.0))
    highway = edge_data.get("highway", "residential")
    narrow = _is_narrow(highway)

    if vehicle_class == "public" and narrow:
        return math.inf  # bus / CNG can't fit narrow roads
    if vehicle_class == "emergency":
        return length    # ambulance / fire bypass narrow penalty and jam

    penalty = 2.2 if (vehicle_class == "private" and narrow) else 1.0
    return length * penalty * (1.0 + traffic_level)


def reverse_dijkstra(
    graph,
    goal: int,
    vehicle_class: str,
    traffic_level: float,
    horizon_meters: float = DEFAULT_HORIZON_METERS,
) -> dict[int, float]:
   
    dist: dict[int, float] = {goal: 0.0}
    heap: list[tuple[float, int]] = [(0.0, goal)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > horizon_meters:
            break
        if d > dist.get(u, math.inf):
            continue
        sources = graph.predecessors(u) if graph.is_directed() else graph.neighbors(u)
        for v in sources:
            edge = _min_edge(graph, v, u)
            if edge is None:
                continue
            w = _edge_weight(edge, vehicle_class, traffic_level)
            if not math.isfinite(w):
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist



@dataclass
class GasWorld:
    graph: Any
    stations: list[Station]
    users: list[User]
    time_slot: str
    seed: int
    # dist[vehicle_class][station_id] -> {node_id: meters}
    dist: dict[str, dict[int, dict[int, float]]] = field(default_factory=dict)

    def distance(self, user: User, station: Station) -> float:
        return self.dist[user.vehicle_class][station.id].get(user.node, math.inf)

    def reachable(self, user: User, station: Station) -> bool:
        """U1: jam + road-size aware reachability."""
        return self.distance(user, station) <= user.fuel_left_km * 1000.0

    def __repr__(self) -> str:
        return (
            f"GasWorld(stations={len(self.stations)}, users={len(self.users)}, "
            f"time_slot={self.time_slot!r}, seed={self.seed})"
        )


def build_world(
    n_users: int,
    time_slot: str = "10:00-12:00",
    seed: int = 42,
    bbox: tuple[float, float, float, float] | None = None,
    max_stations: int | None = None,
    horizon_meters: float | None = None,
    area_name: str | None = None,
    horizon_safety_margin_m: float = 500.0,
) -> GasWorld:
    
    graph = load_dhaka_graph()
    print(f"[gas_world] Graph: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges")

    area_zone_code = AREA_ZONE_CODE.get(area_name) if area_name else None

    stations = load_stations_from_osm(graph, bbox=bbox, max_stations=max_stations,
                                       seed=seed, area_zone_code=area_zone_code)
    configure_stations(stations, seed=seed)
    users = generate_users(graph, n_users=n_users, seed=seed, bbox=bbox,
                            area_zone_code=area_zone_code)

    
    if horizon_meters is None:
        max_fuel_km = max((u.fuel_left_km for u in users), default=12.0)
        horizon_meters = max_fuel_km * 1000.0 + horizon_safety_margin_m
        print(
            f"[gas_world] Horizon derived from max user fuel "
            f"({max_fuel_km:.1f} km) + {horizon_safety_margin_m:.0f} m safety "
            f"→ {horizon_meters/1000:.2f} km"
        )

    traffic_level = _base_traffic_level(time_slot)
    print(
        f"[gas_world] Building distance oracle "
        f"(time_slot={time_slot}, traffic_level={traffic_level}, "
        f"horizon={horizon_meters/1000:.2f}km)..."
    )

    dist: dict[str, dict[int, dict[int, float]]] = {vc: {} for vc in VEHICLE_CLASSES}
    for vc in VEHICLE_CLASSES:
        for s in stations:
            dist[vc][s.id] = reverse_dijkstra(
                graph, s.node, vc, traffic_level, horizon_meters=horizon_meters
            )
        cached_nodes = sum(len(d) for d in dist[vc].values())
        print(
            f"[gas_world]   vehicle_class={vc:9s}: "
            f"{len(stations)} stations, {cached_nodes:,} node entries"
        )

    return GasWorld(
        graph=graph,
        stations=stations,
        users=users,
        time_slot=time_slot,
        seed=seed,
        dist=dist,
    )


if __name__ == "__main__":
   
    bbox = (23.735, 23.770, 90.360, 90.395)

    world = build_world(
        n_users=15,
        time_slot="10:00-12:00",
        seed=42,
        bbox=bbox,
        max_stations=8,
    )
    print(f"\n[summary] {world}")

    # How feasible is each user?
    print("\n[per-user reachability]")
    for u in world.users:
        reachable = sum(1 for s in world.stations if world.reachable(u, s))
        print(
            f"  user {u.id:2d} | class={u.vehicle_class:9s} | zone={u.zone} | "
            f"fuel={u.fuel_left_km:4.1f}km | reaches {reachable}/{len(world.stations)} stations"
        )

    print("\n[per-station info]")
    for s in world.stations:
        print(
            f"  station {s.id:2d} | {s.name[:30]:30s} | zone={s.zone} | "
            f"cap={s.capacity_liters:6.0f}L | cng={s.is_cng} | emerg_only={s.is_emergency_only}"
        )
