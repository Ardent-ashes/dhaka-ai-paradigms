"""
visualize.py — maps + charts for the CSP results
=================================================

Reuses Lab 1's plot style (white bg, light grey edges, color palette).
Outputs go to cache/plots/ to match Lab 1's convention.

Plots produced:
  * plot_assignment_map(world, assignment, unassignable, path) — the headline map
  * plot_station_loads(world, assignment, path) — bar chart of demand vs capacity
  * plot_convergence(convergence, path) — min-conflicts curve
  * plot_comparison_bars(results_by_label, path, ylabel) — generic bar chart
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox

from gas_world import GasWorld, Station, User


# ---------------------------------------------------------------------------
# style constants (lifted from Lab 1)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#17becf", "#bcbd22", "#7f7f7f",
    "#1f77b4", "#ff9896", "#98df8a", "#c5b0d5", "#9467bd",
]


def _ensure_out(output_path: str) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. The assignment map
# ---------------------------------------------------------------------------

def plot_assignment_map(
    world: GasWorld,
    assignment: dict[int, int],
    unassignable: list[int] | None = None,
    title: str = "User → Station assignment",
    output_path: str = "cache/plots/assignment_map.png",
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Render stations, users, and assignment lines on the Dhaka map."""
    unassignable = list(unassignable or [])
    out = _ensure_out(output_path)

    # base map
    fig, ax = ox.plot_graph(
        world.graph,
        node_size=0,
        bgcolor="white",
        edge_color="#c7cedb",
        edge_linewidth=0.55,
        show=False, close=False,
    )

    # assignment lines (drawn first so dots are on top)
    station_color = {s.id: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(world.stations)}
    for uid, sid in assignment.items():
        if sid is None:
            continue
        u = world.users[uid]
        s = world.stations[sid]
        ax.plot(
            [u.lon, s.lon],
            [u.lat, s.lat],
            color=station_color[sid],
            linewidth=1.1,
            alpha=0.75,
            zorder=2,
        )

    # users: assigned (filled), unassignable (red x)
    assigned_uids = set(assignment.keys())
    for u in world.users:
        if u.id in unassignable:
            ax.scatter(u.lon, u.lat, marker="x", s=42, color="#d62728", zorder=4, linewidths=1.6)
        elif u.id in assigned_uids:
            ax.scatter(
                u.lon, u.lat,
                marker="o", s=28,
                color=station_color[assignment[u.id]],
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
        else:
            # in-domain but unassigned (rare)
            ax.scatter(u.lon, u.lat, marker="o", s=22, facecolors="none", edgecolors="black", zorder=4)

    # stations: fuel-pump marker
    for s in world.stations:
        ax.scatter(
            s.lon, s.lat,
            marker="s", s=120,
            color=station_color[s.id],
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            f"S{s.id}",
            (s.lon, s.lat),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="black", zorder=6,
        )

    # legend
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Station"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=8, label="User (assigned)"),
        Line2D([0], [0], marker="x", color="#d62728", markersize=9,
               linestyle="", label="User (infeasible)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95,
              facecolor="white", edgecolor="#c9d1d9")

    # zoom to bbox or to populated area
    if bbox is None:
        # autocompute from stations + users
        lats = [s.lat for s in world.stations] + [u.lat for u in world.users]
        lons = [s.lon for s in world.stations] + [u.lon for u in world.users]
        if lats and lons:
            pad_lat = 0.005
            pad_lon = 0.005
            bbox = (min(lats) - pad_lat, max(lats) + pad_lat,
                    min(lons) - pad_lon, max(lons) + pad_lon)
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    ax.set_title(title, fontsize=10)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved map: {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Station loads bar chart
# ---------------------------------------------------------------------------

def plot_station_loads(
    world: GasWorld,
    assignment: dict[int, int],
    policy_strategic_frac: float = 0.10,
    policy_public_frac: float = 0.60,
    title: str = "Station load vs capacity",
    output_path: str = "cache/plots/station_loads.png",
) -> Path:
    out = _ensure_out(output_path)
    # accumulate demand per station
    loads = {s.id: {"private": 0.0, "public": 0.0, "emergency": 0.0} for s in world.stations}
    for uid, sid in assignment.items():
        if sid is None:
            continue
        u = world.users[uid]
        loads[sid][u.vehicle_class] += u.demand_liters

    s_ids = [s.id for s in world.stations]
    priv = [loads[i]["private"] for i in s_ids]
    pub = [loads[i]["public"] for i in s_ids]
    emerg = [loads[i]["emergency"] for i in s_ids]
    caps = [s.capacity_liters for s in world.stations]
    cap_civ = [c * (1.0 - policy_strategic_frac) for c in caps]
    priv_caps = [c * (1.0 - policy_public_frac) for c in cap_civ]

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(s_ids) + 4), 5))
    x = list(range(len(s_ids)))
    ax.bar(x, priv, color="#377eb8", label="private")
    ax.bar(x, pub, bottom=priv, color="#4daf4a", label="public")
    ax.bar(x, emerg, bottom=[p + q for p, q in zip(priv, pub)], color="#e41a1c", label="emergency")
    ax.plot(x, caps, "k_", markersize=18, label="total cap")
    ax.plot(x, cap_civ, "0.4", marker="_", markersize=14, linestyle="", label="civilian cap")
    ax.plot(x, priv_caps, "0.6", marker="_", markersize=10, linestyle="", label="private cap")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in s_ids])
    ax.set_ylabel("Liters")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[viz] Saved loads: {out}")
    return out


# ---------------------------------------------------------------------------
# 3. Min-Conflicts convergence curve
# ---------------------------------------------------------------------------

def plot_convergence(
    convergence: list[int],
    title: str = "Min-Conflicts convergence",
    output_path: str = "cache/plots/convergence.png",
) -> Path:
    out = _ensure_out(output_path)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(convergence, color="#1f77b4", linewidth=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Violations")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[viz] Saved convergence: {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Generic comparison bar chart (heuristic-vs-heuristic, run-A style)
# ---------------------------------------------------------------------------

def plot_comparison_bars(
    results_by_label: dict[str, float],
    title: str,
    ylabel: str,
    output_path: str,
    color: str = "#4c78a8",
) -> Path:
    out = _ensure_out(output_path)
    labels = list(results_by_label.keys())
    values = [results_by_label[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels) + 3), 4.5))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(values):
        if math.isfinite(v):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[viz] Saved comparison: {out}")
    return out


# ---------------------------------------------------------------------------
# 5. Scaling plot (Run B): runtime vs N
# ---------------------------------------------------------------------------

def plot_scaling(
    n_values: list[int],
    series_by_label: dict[str, list[float]],
    ylabel: str = "Runtime (ms)",
    title: str = "Scaling — runtime vs #users",
    output_path: str = "cache/plots/scaling.png",
    log_y: bool = True,
) -> Path:
    out = _ensure_out(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, ys) in enumerate(series_by_label.items()):
        ax.plot(n_values, ys, marker="o", label=label, color=_PALETTE[i % len(_PALETTE)])
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("# users")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[viz] Saved scaling plot: {out}")
    return out


if __name__ == "__main__":
    # End-to-end smoke test: build world, solve both ways, render everything.
    from gas_world import build_world
    from csp_model import build_csp, Policy
    from backtracking import backtracking_search
    from min_conflicts import min_conflicts

    bbox = (23.735, 23.770, 90.360, 90.395)
    world = build_world(n_users=15, time_slot="10:00-12:00", seed=42, bbox=bbox, max_stations=8)
    policy = Policy(today_parity="any", min_service_quota=0)
    csp = build_csp(world, policy=policy)

    bt = backtracking_search(csp, use_mrv=True, use_lcv=True, use_fc=True)
    print(f"\n{bt.summary()}")
    plot_assignment_map(world, bt.assignment, bt.unassignable,
                        title=f"Backtracking (cost={bt.cost:.0f}, served={len(bt.assignment)}/{len(world.users)})",
                        output_path="cache/plots/bt_map.png", bbox=bbox)
    plot_station_loads(world, bt.assignment, output_path="cache/plots/bt_loads.png")

    mc = min_conflicts(csp, max_steps=2000, max_restarts=3, seed=42)
    print(f"{mc.summary()}")
    plot_assignment_map(world, mc.assignment, mc.unassignable,
                        title=f"Min-Conflicts (cost={mc.cost:.0f}, served={len(mc.assignment)}/{len(world.users)})",
                        output_path="cache/plots/mc_map.png", bbox=bbox)
    plot_station_loads(world, mc.assignment, output_path="cache/plots/mc_loads.png")
    plot_convergence(mc.convergence, output_path="cache/plots/mc_convergence.png")

    plot_comparison_bars(
        {"Backtracking": bt.cost, "Min-Conflicts": mc.cost},
        title="Total cost (lower is better)",
        ylabel="Cost",
        output_path="cache/plots/cost_compare.png",
    )
    plot_comparison_bars(
        {"Backtracking": bt.runtime_ms, "Min-Conflicts": mc.runtime_ms},
        title="Runtime",
        ylabel="ms",
        output_path="cache/plots/runtime_compare.png",
    )
    print("\n[viz] All smoke-test plots in cache/plots/")
