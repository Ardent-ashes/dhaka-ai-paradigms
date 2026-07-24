from pathlib import Path

import osmnx as ox


def load_dhaka_graph(
    cache_path: str = "cache/dhaka_all.graphml",
    force_refresh: bool = False,
):
    """Load Dhaka OSM road graph with local cache."""
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force_refresh:
        print(f"[map] Loading cached graph: {cache_file}")
        return ox.load_graphml(cache_file)

    print("[map] Downloading Dhaka OSM graph (network_type='all')...")
    graph = ox.graph_from_place("Dhaka, Bangladesh", network_type="all", simplify=True)
    ox.save_graphml(graph, cache_file)
    print(f"[map] Saved cache to: {cache_file}")
    return graph