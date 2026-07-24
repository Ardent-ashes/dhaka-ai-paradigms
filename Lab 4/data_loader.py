"""Loads the cached Dhaka OSM road graph (re-uses the earlier labs' cache)."""
from pathlib import Path
import osmnx as ox

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "cache" / "dhaka_all.graphml"


def load_dhaka_graph(cache_path: str = None):
    cache_file = Path(cache_path) if cache_path else DEFAULT_CACHE
    if cache_file.exists():
        print(f"[map] Loading cached graph: {cache_file}")
        return ox.load_graphml(cache_file)
    print("[map] Downloading Dhaka OSM graph ...")
    G = ox.graph_from_place("Dhaka, Bangladesh", network_type="drive", simplify=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache_file)
    return G
