from __future__ import annotations

from dataclasses import asdict
import argparse
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import pandas as pd

from cost_engine import Scenario, build_euclidean_practical_cache
from data_loader import load_dhaka_graph
from search_algorithms import (
    astar,
    bfs,
    build_admissible_heuristic_cache,
    dfs,
    greedy_best_first,
    ucs,
    weighted_astar,
)


OUTPUT_DIR = Path("cache/benchmark")


def random_scenario(rng: random.Random) -> Scenario:
    return Scenario(
        vehicle=rng.choice(["walk", "car", "bus"]),
        time_slot=rng.choice(
            ["07:00-10:00", "10:00-12:00", "15:00-17:00", "17:00-21:00", "21:00-06:00"]
        ),
        seed=rng.randint(0, 999999),
        gender=rng.choice(["male", "female"]),
        alone=rng.choice([True, False]),
        pace=rng.choice(["relaxed", "rush"]),
    )


def run_benchmark(count: int = 10, master_seed: int = 2026) -> pd.DataFrame:
    rng = random.Random(master_seed)
    graph = load_dhaka_graph()
    nodes = list(graph.nodes)
    print(f"[benchmark] Graph ready: nodes={len(nodes):,}")

    rows: list[dict] = []
    algo_runs = [
        ("BFS", lambda g, s, t, sc, h: bfs(g, s, t, sc)),
        ("DFS", lambda g, s, t, sc, h: dfs(g, s, t, sc)),
        ("UCS", lambda g, s, t, sc, h: ucs(g, s, t, sc)),
        ("Greedy Best-First", lambda g, s, t, sc, h: greedy_best_first(g, s, t, sc, h_cache=h)),
        ("A*", lambda g, s, t, sc, h: astar(g, s, t, sc, h_cache=h)),
        ("Weighted A*", lambda g, s, t, sc, h: weighted_astar(g, s, t, sc, h_cache=h)),
    ]

    t0 = time.perf_counter()
    for i in range(1, count + 1):
        src = rng.choice(nodes)
        dst = rng.choice(nodes)
        while dst == src:
            dst = rng.choice(nodes)

        scenario = random_scenario(rng)
        # Setup-1 heuristic: admissible-oriented reverse-Dijkstra cache
        h_cache_1 = build_admissible_heuristic_cache(graph, dst, scenario)
        # Setup-2 heuristic: practical Euclidean + road-wise (not guaranteed admissible)
        h_cache_2 = build_euclidean_practical_cache(graph, dst, scenario)
        print(
            f"[benchmark] Scenario {i}/{count} | src={src} dst={dst} "
            f"vehicle={scenario.vehicle} time={scenario.time_slot}"
        )

        scenario_info = asdict(scenario)
        for setup_name, h_cache in [
            ("setup1_admissible_reverse_dijkstra", h_cache_1),
            ("setup2_practical_euclidean", h_cache_2),
        ]:
            for name, runner in algo_runs:
                res = runner(graph, src, dst, scenario, h_cache)
                rows.append(
                    {
                        "scenario_id": i,
                        "setup": setup_name,
                        "algorithm": name,
                        "found": res.found,
                        "total_cost": res.total_cost,
                        "distance_m": res.distance_m,
                        "nodes_expanded": res.nodes_expanded,
                        "nodes_popped": res.nodes_popped,
                        "revisit_count": res.revisit_count,
                        "runtime_ms": res.runtime_ms,
                        **scenario_info,
                    }
                )

    elapsed = time.perf_counter() - t0
    print(f"[benchmark] Finished {count} scenarios in {elapsed:.1f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        df.groupby(["setup", "algorithm"], as_index=False)
        .agg(
            success_rate=("found", "mean"),
            avg_cost=("total_cost", lambda s: s[df.loc[s.index, "found"]].mean()),
            avg_distance_m=("distance_m", lambda s: s[df.loc[s.index, "found"]].mean()),
            avg_expanded=("nodes_expanded", "mean"),
            avg_runtime_ms=("runtime_ms", "mean"),
        )
        .sort_values(["setup", "avg_runtime_ms"])
    )
    avg["success_rate"] = (avg["success_rate"] * 100).round(2)
    return avg


def plot_summary(summary: pd.DataFrame, out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for setup_name, sdf in summary.groupby("setup", as_index=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sdf["algorithm"], sdf["avg_runtime_ms"], color="#4c78a8")
        ax.set_title(f"Average Runtime by Algorithm ({count} scenarios) — {setup_name}")
        ax.set_ylabel("Runtime (ms)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"avg_runtime__{setup_name}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sdf["algorithm"], sdf["avg_expanded"], color="#f58518")
        ax.set_title(f"Average Expanded Nodes by Algorithm ({count} scenarios) — {setup_name}")
        ax.set_ylabel("Expanded Nodes")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"avg_expanded_nodes__{setup_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of random scenarios")
    parser.add_argument("--seed", type=int, default=2026, help="Master RNG seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed_df = run_benchmark(count=args.count, master_seed=args.seed)
    summary_df = summarize(detailed_df)

    detailed_csv = OUTPUT_DIR / f"detailed_runs_{args.count}.csv"
    summary_csv = OUTPUT_DIR / f"summary_{args.count}.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_summary(summary_df, OUTPUT_DIR, count=args.count)

    print(f"\n=== Average Comparison ({args.count} scenarios) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved detailed runs: {detailed_csv}")
    print(f"Saved summary table: {summary_csv}")
    print(f"Saved charts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
