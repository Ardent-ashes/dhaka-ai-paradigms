
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString
import osmnx as ox

from data_loader import load_dhaka_graph

# ---------------------------------------------------------------------------
# Preset study areas (centre lat, lon). Radius chosen at build time.
# ---------------------------------------------------------------------------
AREAS = {
    "dhanmondi": (23.7461, 90.3742),
    "gulshan":   (23.7925, 90.4078),
    "mirpur":    (23.8041, 90.3667),
    "old_dhaka": (23.7104, 90.4074),
    "uttara":    (23.8759, 90.3795),
}

# Road importance by OSM 'highway' tag (bigger road -> more important to watch).
IMPORTANCE = {
    "motorway": 3.0, "motorway_link": 2.5,
    "trunk": 3.0, "trunk_link": 2.5,
    "primary": 3.0, "primary_link": 2.5,
    "secondary": 2.0, "secondary_link": 1.8,
    "tertiary": 1.5, "tertiary_link": 1.3,
    "residential": 1.0, "unclassified": 1.0,
    "living_street": 0.8, "service": 0.5,
}
DEFAULT_IMPORTANCE = 1.0

HOTSPOT_TAGS = {
    "amenity": ["school", "college", "university", "marketplace", "hospital",
                "bank", "bus_station", "place_of_worship", "police"],
    "shop": True,
}


def road_importance(highway):
    if isinstance(highway, list):
        return max(IMPORTANCE.get(h, DEFAULT_IMPORTANCE) for h in highway)
    return IMPORTANCE.get(highway, DEFAULT_IMPORTANCE)


@dataclass
class World:
    Hp: object                       
    dx: np.ndarray                   
    dy: np.ndarray                   
    dw: np.ndarray                   
    cand_xy: np.ndarray              
    cand_ids: list                   
    neigh: list                      
    max_R: float
    bbox: tuple                     
    crime: np.ndarray = None         
    overlap_penalty: float = 0.0     
    _cand_tree: cKDTree = field(default=None, repr=False)

    # ---- derived helpers -------------------------------------------------
    @property
    def total_weight(self):
        return float(self.dw.sum())

    def snap(self, x, y):
        """Nearest candidate index to a continuous (x,y) -- used by PSO."""
        _, i = self._cand_tree.query([x, y])
        return int(i)

    def camera_cover_idx(self, cand_index, theta, R, half_fov):
        """Indices of demand points seen by one camera (cone model)."""
        idx, dist, brg = self.neigh[cand_index]
        if idx.size == 0:
            return idx
        ang = np.abs((brg - theta + np.pi) % (2 * np.pi) - np.pi)
        mask = (dist <= R) & (ang <= half_fov)
        return idx[mask]

    def covered_mask(self, cameras):
        """Boolean array over demand points covered by a set of cameras.
        cameras: list of (cand_index, theta, R, half_fov)."""
        covered = np.zeros(self.dw.size, dtype=bool)
        for c, th, R, hf in cameras:
            covered[self.camera_cover_idx(c, th, R, hf)] = True
        return covered

    def coverage_weight(self, cameras):
        return float(self.dw[self.covered_mask(cameras)].sum())

    def coverage_pct(self, cameras):
        return 100.0 * self.coverage_weight(cameras) / self.total_weight

    def coverage_count(self, cameras):
        """How many cameras see each demand point (for overlap)."""
        cnt = np.zeros(self.dw.size, dtype=int)
        for c, th, R, hf in cameras:
            cnt[self.camera_cover_idx(c, th, R, hf)] += 1
        return cnt

    def overlap_weight(self, cameras):
        """Weighted redundant coverage = sum weight*(times_seen - 1)."""
        redund = np.clip(self.coverage_count(cameras) - 1, 0, None)
        return float((self.dw * redund).sum())

    def overlap_pct(self, cameras):
        return 100.0 * self.overlap_weight(cameras) / self.total_weight

    def objective(self, cameras):
        """What the solvers MAXIMISE: coverage minus overlap penalty.
        With overlap_penalty=0 this equals plain coverage (submodular)."""
        cnt = self.coverage_count(cameras)
        covered = float(self.dw[cnt >= 1].sum())
        overlap = float((self.dw * np.clip(cnt - 1, 0, None)).sum())
        return covered - self.overlap_penalty * overlap


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------
def _subgraph_bbox(G, lat, lon, radius_m):
    """Crop the full graph to a lat/lon box around a centre point."""
    # ~metres-per-degree at Dhaka's latitude
    dlat = radius_m / 111_000.0
    dlon = radius_m / (111_000.0 * np.cos(np.deg2rad(lat)))
    north, south = lat + dlat, lat - dlat
    east, west = lon + dlon, lon - dlon
    keep = [n for n, d in G.nodes(data=True)
            if south <= d["y"] <= north and west <= d["x"] <= east]
    return G.subgraph(keep).copy(), (north, south, east, west)


def _fetch_hotspots(lat, lon, radius_m, target_crs):
    """Return projected (x,y) array of hotspot POIs, or empty on failure."""
    try:
        gdf = ox.features_from_point((lat, lon), tags=HOTSPOT_TAGS, dist=radius_m)
        if gdf.empty:
            return np.empty((0, 2))
        gdf = gdf.to_crs(target_crs)
        cent = gdf.geometry.centroid
        return np.column_stack([cent.x.values, cent.y.values])
    except Exception as e:               # offline / overpass down -> graceful
        print(f"[world] POI fetch skipped ({type(e).__name__}); no hotspots used.")
        return np.empty((0, 2))


def _crime_layer(dx, dy, pois, bbox, n_clusters, cmax, seed):
    """Synthetic crime / crowd-density score per demand point.

    Clusters are placed half on real POIs
    (crowded = riskier) and half at random locations, each spreading a Gaussian
    risk.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bbox
    centers, intens, radii = [], [], []

    n_poi = min(n_clusters // 2, pois.shape[0]) if pois.shape[0] else 0
    if n_poi:
        for s in rng.choice(pois.shape[0], n_poi, replace=False):
            centers.append(pois[s])
            intens.append(rng.uniform(0.6, 1.0) * cmax)
            radii.append(rng.uniform(160, 340))
    for _ in range(n_clusters - n_poi):
        centers.append([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        intens.append(rng.uniform(0.4, 1.0) * cmax)
        radii.append(rng.uniform(160, 340))

    score = np.zeros(dx.size)
    for (cx, cy), a, r in zip(centers, intens, radii):
        d2 = (dx - cx) ** 2 + (dy - cy) ** 2
        score += a * np.exp(-d2 / (2 * r * r))
    return np.clip(score, 0, cmax)


def build_world(area="dhanmondi", radius_m=1200, spacing=25.0, max_R=250.0,
                intersection_weight=40.0, hotspot=True, hotspot_radius=45.0,
                hotspot_mult=2.5, crime=True, crime_clusters=8, crime_max=9.0,
                crime_scale=4.0, crime_seed=7, overlap_penalty=0.0,
                min_street_count=3, max_candidates=400, seed=42,
                verbose=True, G=None):
    lat, lon = AREAS[area]
    G = G if G is not None else load_dhaka_graph()
    H, _ = _subgraph_bbox(G, lat, lon, radius_m)
    if H.number_of_nodes() == 0:
        raise ValueError("No roads in that area/radius.")
    Hp = ox.project_graph(H)
    crs = Hp.graph["crs"]
    if verbose:
        print(f"[world] {area}: {Hp.number_of_nodes()} nodes, "
              f"{Hp.number_of_edges()} directed edges (r={radius_m} m)")

    # ---- demand points from road segments -------------------------------
    dx, dy, dw = [], [], []
    seen = set()
    for u, v, data in Hp.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        if "geometry" in data and isinstance(data["geometry"], LineString):
            line = data["geometry"]
        else:
            line = LineString([(Hp.nodes[u]["x"], Hp.nodes[u]["y"]),
                               (Hp.nodes[v]["x"], Hp.nodes[v]["y"])])
        length = float(data.get("length", line.length))
        imp = road_importance(data.get("highway"))
        n = max(1, int(length // spacing))
        seg_w = imp * (length / (n + 1))          # weight per sample point
        for i in range(n + 1):
            p = line.interpolate(i / n if n else 0.5, normalized=True)
            dx.append(p.x); dy.append(p.y); dw.append(seg_w)

    # ---- demand points from intersections (extra weight by degree) ------
    for nid, d in Hp.nodes(data=True):
        sc = d.get("street_count", 0) or 0
        if sc >= min_street_count:
            dx.append(d["x"]); dy.append(d["y"])
            dw.append(intersection_weight * sc)

    dx = np.asarray(dx); dy = np.asarray(dy); dw = np.asarray(dw, float)
    dbbox = (dx.min(), dx.max(), dy.min(), dy.max())

    # POIs fetched once, shared by the hotspot bonus and the crime layer
    pois = _fetch_hotspots(lat, lon, radius_m, crs) if (hotspot or crime) \
        else np.empty((0, 2))

    # ---- hotspot bonus (real OSM POIs) 
    if hotspot and pois.shape[0] > 0:
        near, _ = cKDTree(pois).query(np.column_stack([dx, dy]),
                                      distance_upper_bound=hotspot_radius)
        hit = np.isfinite(near)
        dw[hit] *= hotspot_mult
        if verbose:
            print(f"[world] {pois.shape[0]} POIs -> {int(hit.sum())} "
                  f"demand points boosted x{hotspot_mult}")

    # ---- synthetic crime / crowd-density layer 
    crime_score = None
    if crime:
        crime_score = _crime_layer(dx, dy, pois, dbbox, crime_clusters,
                                   crime_max, crime_seed)
        dw *= (1 + crime_score / crime_scale)          
        if verbose:
            hi = int((crime_score > 0.5 * crime_max).sum())
            print(f"[world] crime layer: {crime_clusters} clusters, "
                  f"{hi} demand points in high-risk zones")

    # ---- candidate camera locations (intersections) 
    cands = [(nid, d) for nid, d in Hp.nodes(data=True)
             if (d.get("street_count", 0) or 0) >= min_street_count]
    cands.sort(key=lambda t: t[1].get("street_count", 0), reverse=True)
    cands = cands[:max_candidates]
    cand_ids = [nid for nid, _ in cands]
    cand_xy = np.array([[d["x"], d["y"]] for _, d in cands])
    if verbose:
        print(f"[world] {len(dw)} demand points | {len(cand_ids)} camera candidates")

    # ---- precompute coverage neighbourhoods -----------------------------
    dtree = cKDTree(np.column_stack([dx, dy]))
    neigh = []
    for cx, cy in cand_xy:
        idx = np.array(dtree.query_ball_point([cx, cy], max_R), dtype=int)
        if idx.size:
            dxi, dyi = dx[idx] - cx, dy[idx] - cy
            dist = np.hypot(dxi, dyi)
            brg = np.arctan2(dyi, dxi)
        else:
            dist = np.array([]); brg = np.array([])
        neigh.append((idx, dist, brg))

    return World(Hp=Hp, dx=dx, dy=dy, dw=dw, cand_xy=cand_xy, cand_ids=cand_ids,
                 neigh=neigh, max_R=max_R,
                 bbox=(cand_xy[:, 0].min(), cand_xy[:, 0].max(),
                       cand_xy[:, 1].min(), cand_xy[:, 1].max()),
                 crime=crime_score, overlap_penalty=overlap_penalty,
                 _cand_tree=cKDTree(cand_xy))
