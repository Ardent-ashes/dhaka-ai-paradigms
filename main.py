from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox

from analysis import plot_metrics, print_results_table, save_results_csv
from cost_engine import Scenario, build_euclidean_practical_cache
from data_loader import load_dhaka_graph
from search_algorithms import (
    astar,
    bfs,
    build_admissible_heuristic_cache,
    dfs,
    greedy_best_first,
    iddfs,
    ucs,
    weighted_astar,
)


def _ask_bounded_float(prompt, default, low, high):
    while True:
        raw = input(f"{prompt} [{default}] (range: {low} to {high}): ").strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("[input] Please enter a numeric value.")
            continue
        if low <= value <= high:
            return value
        print(f"[input] Value must be between {low} and {high}.")


def _ask_int(prompt, default, low, high):
    while True:
        raw = input(f"{prompt} [{default}] (range: {low} to {high}): ").strip()
        if not raw:
            return int(default)
        try:
            value = int(raw)
        except ValueError:
            print("[input] Please enter an integer value.")
            continue
        if low <= value <= high:
            return value
        print(f"[input] Value must be between {low} and {high}.")


def _ask_menu(prompt, options, default_key):
    print(prompt)
    for key, label in options.items():
        marker = " (default)" if key == default_key else ""
        print(f"  {key}. {label}{marker}")
    while True:
        raw = input("Choose option number: ").strip()
        if not raw:
            return options[default_key]
        if raw in options:
            return options[raw]
        print(f"[input] Invalid choice. Choose one of: {', '.join(options.keys())}")


def _plot_best_route(graph, results, run_id):
    found_results = [r for r in results if r.found]
    if not found_results:
        print("[route] No route found by any algorithm.")
        return
    best = min(found_results, key=lambda r: r.total_cost)
    output = Path("cache/plots")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"best_route.png"
    fig, _ = ox.plot_graph_route(
        graph,
        best.path,
        route_linewidth=3,
        route_alpha=0.9,
        route_color="red",
        node_size=0,
        bgcolor="white",
        edge_color="#c7cedb",
        edge_linewidth=0.6,
        show=False,
        close=False,
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[route] Best route ({best.algorithm}) plot: {path}")


def _plot_all_algorithm_routes(graph, results, run_id):
    """One map: every algorithm that found a path, different colors + legend."""
    entries = [(r.algorithm, r.path) for r in results if r.found and len(r.path) >= 2]
    if not entries:
        print("[route] No paths to draw for combined map.")
        return
    names, routes = zip(*entries)
    palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
    ]
    route_colors = [palette[i % len(palette)] for i in range(len(routes))]
    route_linewidths = [2.8] * len(routes)

    output = Path("cache/plots")
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"all_routes.png"

    fig, ax = ox.plot_graph_routes(
        graph,
        list(routes),
        route_colors=route_colors,
        route_linewidths=route_linewidths,
        route_alpha=0.72,
        node_size=0,
        bgcolor="white",
        edge_color="#c7cedb",
        edge_linewidth=0.55,
        show=False,
        close=False,
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    legend_elements = [
        Line2D([0], [0], color=route_colors[i], lw=3, label=names[i]) for i in range(len(names))
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=7,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#c9d1d9",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[route] All algorithms overlay: {out_path}")


def run_all_algorithms(graph, start_node, end_node, scenario, h_cache):
    results = []
    runners = [
        ("BFS", lambda: bfs(graph, start_node, end_node, scenario)),
        ("DFS", lambda: dfs(graph, start_node, end_node, scenario)),
        ("IDDFS", lambda: iddfs(graph, start_node, end_node, scenario)),
        ("UCS", lambda: ucs(graph, start_node, end_node, scenario)),
        (
            "Greedy Best-First",
            lambda: greedy_best_first(
                graph, start_node, end_node, scenario, h_cache=h_cache
            ),
        ),
        ("A*", lambda: astar(graph, start_node, end_node, scenario, h_cache=h_cache)),
        (
            "Weighted A*",
            lambda: weighted_astar(
                graph, start_node, end_node, scenario, h_cache=h_cache
            ),
        ),
    ]
    for name, run in runners:
        print(f"[run] {name}...")
        results.append(run())
    return results


def main():
    print("Loading Dhaka graph...")
    graph = load_dhaka_graph()
    print(f"[map] Nodes={len(graph.nodes):,}, Edges={len(graph.edges):,}")

    while True:
        print("\n=== New Scenario ===")
        print("[hint] Approx Dhaka latitude range: 23.60 to 23.95")
        print("[hint] Approx Dhaka longitude range: 90.20 to 90.55")
        src_lat = _ask_bounded_float("Source latitude", 23.8103, 23.60, 23.95)
        src_lon = _ask_bounded_float("Source longitude", 90.4125, 90.20, 90.55)
        dst_lat = _ask_bounded_float("Destination latitude", 23.7509, 23.60, 23.95)
        dst_lon = _ask_bounded_float("Destination longitude", 90.3931, 90.20, 90.55)

        vehicle = _ask_menu(
            "\nSelect vehicle type (used by edge_cost for road-access/penalty):",
            {"1": "walk", "2": "car", "3": "bus"},
            default_key="1",
        )
        time_slot = _ask_menu(
            "\nSelect time slot (used by edge_cost for traffic level):",
            {
                "1": "07:00-10:00",
                "2": "10:00-12:00",
                "3": "15:00-17:00",
                "4": "17:00-21:00",
                "5": "21:00-06:00",
            },
            default_key="4",
        )
        seed = _ask_int(
            "Traffic random seed (same seed => same random jam pattern)",
            42,
            0,
            999999,
        )

        gender = _ask_menu(
            "\nGender (affects safety term in real edge_cost only, not heuristic):",
            {"1": "male", "2": "female"},
            default_key="1",
        )
        alone = _ask_menu(
            "\nTravelling alone? (affects safety term in real edge_cost only):",
            {"1": "yes", "2": "no"},
            default_key="2",
        ) == "yes"
        pace = _ask_menu(
            "\nPace (rush = more traffic stress; affects real edge_cost + safety sensitivity):",
            {"1": "relaxed", "2": "rush"},
            default_key="1",
        )
        weight_profile = _ask_menu(
            "\nWeight profile (normalized priorities):",
            {
                "1": "balanced",
                "2": "fastest",
                "3": "safest",
                "4": "distance",
                "5": "custom",
            },
            default_key="1",
        )

        distance_priority = vehicle_priority = traffic_priority = safety_priority = accident_priority = 0.0
        if weight_profile == "custom":
            print("\nSet custom priorities (0 to 10). Higher means more importance.")
            distance_priority = _ask_bounded_float("Distance priority", 3, 0, 10)
            vehicle_priority = _ask_bounded_float("Vehicle-feasibility priority", 2, 0, 10)
            traffic_priority = _ask_bounded_float("Traffic priority", 3, 0, 10)
            safety_priority = _ask_bounded_float("Safety priority", 2, 0, 10)
            accident_priority = _ask_bounded_float("Accident priority", 2, 0, 10)

        scenario = Scenario(
            vehicle=vehicle,
            time_slot=time_slot,
            seed=seed,
            gender=gender,
            alone=alone,
            pace=pace,
            weight_profile=weight_profile,
            distance_priority=distance_priority,
            vehicle_priority=vehicle_priority,
            traffic_priority=traffic_priority,
            safety_priority=safety_priority,
            accident_priority=accident_priority,
        )
        print(
            f"[scenario] vehicle={scenario.vehicle}, time_slot={scenario.time_slot}, seed={scenario.seed}, "
            f"gender={scenario.gender}, alone={scenario.alone}, pace={scenario.pace}, "
            f"weight_profile={scenario.weight_profile}"
        )
        print(
            "[relation] edge_cost() uses vehicle + traffic + jam + gender/alone/pace safety. "
            "Heuristic uses a deterministic lower-bound of edge_cost for admissible A*."
        )

        start_node = ox.distance.nearest_nodes(graph, X=src_lon, Y=src_lat)
        end_node = ox.distance.nearest_nodes(graph, X=dst_lon, Y=dst_lat)
        print(f"[nodes] start={start_node}, end={end_node}")

        print(
            "[heuristic] Building admissible h(n) cache = lower-bound edge cost "
            "(no random jam and no user-specific safety term)."
        )
        h_cache = build_admissible_heuristic_cache(graph, end_node, scenario)
        print(
            f"[heuristic] h(start)≈{h_cache.get(start_node, float('inf')):.2f} "
            "(edge-wise lower bound <= real edge_cost, so A* heuristic stays admissible)"
        )

        print("[runset] Setup-1: admissible heuristic")
        results = run_all_algorithms(graph, start_node, end_node, scenario, h_cache)
        print_results_table(results)

        print(
            "\n[heuristic] Building Setup-2 cache = Euclidean + local road-wise jam/safety "
            
        )
        h_cache_alt = build_euclidean_practical_cache(graph, end_node, scenario)
        print(
            f"[heuristic] setup-2 h(start)≈{h_cache_alt.get(start_node, float('inf')):.2f}"
        )
        print("[runset] Setup-2: Euclidean practical heuristic (non-admissible)")
        results_alt = run_all_algorithms(graph, start_node, end_node, scenario, h_cache_alt)
        print_results_table(results_alt)

        # Keep plotting/logging tied to setup-1 for stable trend tracking.
        results = results
        save_results_csv(results, scenario=scenario)
        plot_metrics(results)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _plot_best_route(graph, results, run_id)
        _plot_all_algorithm_routes(graph, results, run_id)

        again = _ask_menu("\nRun another scenario?", {"1": "yes", "2": "no"}, default_key="1")
        if again != "yes":
            print("Finished.")
            break


if __name__ == "__main__":
    main()