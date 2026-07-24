from pathlib import Path
import csv

import matplotlib.pyplot as plt
import pandas as pd


def print_results_table(results):
    rows = []
    for r in results:
        rows.append(
            [
                r.algorithm,
                "Yes" if r.found else "No",
                f"{r.total_cost:.2f}" if r.found else "INF",
                f"{r.distance_m:.1f}",
                str(r.nodes_expanded),
                str(r.nodes_popped),
                str(r.revisit_count),
                f"{r.runtime_ms:.2f}",
            ]
        )

    headers = [
        "Algorithm",
        "Found",
        "Cost",
        "Distance(m)",
        "Expanded",
        "Popped",
        "Revisits",
        "Runtime(ms)",
    ]
    width = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]
    line = "".join(h.ljust(width[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("".join(val.ljust(width[i]) for i, val in enumerate(row)))


def save_results_csv(results, csv_path="cache/run_history.csv", scenario=None):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "vehicle",
                    "time_slot",
                    "seed",
                    "gender",
                    "alone",
                    "pace",
                    "algorithm",
                    "found",
                    "total_cost",
                    "distance_m",
                    "nodes_expanded",
                    "nodes_popped",
                    "revisit_count",
                    "runtime_ms",
                ]
            )
        for r in results:
            writer.writerow(
                [
                    getattr(scenario, "vehicle", "na"),
                    getattr(scenario, "time_slot", "na"),
                    getattr(scenario, "seed", -1),
                    getattr(scenario, "gender", "na"),
                    getattr(scenario, "alone", "na"),
                    getattr(scenario, "pace", "na"),
                    r.algorithm,
                    r.found,
                    r.total_cost,
                    r.distance_m,
                    r.nodes_expanded,
                    r.nodes_popped,
                    r.revisit_count,
                    r.runtime_ms,
                ]
            )
    print(f"[log] Saved run metrics to {path}")


def plot_metrics(results, output_dir="cache/plots"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "algorithm": r.algorithm,
                "runtime_ms": r.runtime_ms,
                "nodes_expanded": r.nodes_expanded,
                "total_cost": r.total_cost if r.found else None,
            }
            for r in results
        ]
    )

    charts = [
        ("runtime_ms", "Runtime (ms)", "runtime_ms.png"),
        ("nodes_expanded", "Expanded Nodes", "expanded_nodes.png"),
        ("total_cost", "Total Cost", "total_cost.png"),
    ]
    for col, ylabel, filename in charts:
        plt.figure(figsize=(10, 5))
        plt.bar(df["algorithm"], df[col].fillna(0))
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Algorithm")
        plt.tight_layout()
        target = out / filename
        plt.savefig(target, dpi=140)
        plt.close()
        print(f"[plot] {target}")
