"""Loads the Dhaka OSM road graph (re-uses the cached graph from Lab/Lab2)."""
from pathlib import Path
import osmnx as ox

# The full-Dhaka graph was already downloaded & cached by the earlier labs.
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "cache" / "dhaka_all.graphml"


def load_dhaka_graph(cache_path: str = None, force_refresh: bool = False):
    """Load Dhaka OSM road graph with local cache."""
    cache_file = Path(cache_path) if cache_path else DEFAULT_CACHE
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force_refresh:
        print(f"[map] Loading cached graph: {cache_file}")
        return ox.load_graphml(cache_file)

    print("[map] Downloading Dhaka OSM graph (network_type='all')...")
    graph = ox.graph_from_place("Dhaka, Bangladesh", network_type="all", simplify=True)
    ox.save_graphml(graph, cache_file)
    print(f"[map] Saved cache to: {cache_file}")
    return graph
