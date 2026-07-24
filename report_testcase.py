"""Generate concrete Assignment-1 test-case figures + a per-query metrics table
for the lab report. Non-interactive."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox

from cost_engine import Scenario
from data_loader import load_dhaka_graph
from search_algorithms import (astar, bfs, ucs, greedy_best_first, weighted_astar,
                               build_admissible_heuristic_cache)

FIG = Path("Lab Report ") / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# A concrete test query: Uttara -> Dhanmondi, evening peak.
SRC = (23.8103, 90.4125)
DST = (23.7509, 90.3931)

print("loading graph ...")
G = load_dhaka_graph()
s = ox.distance.nearest_nodes(G, X=SRC[1], Y=SRC[0])
t = ox.distance.nearest_nodes(G, X=DST[1], Y=DST[0])
print("start", s, "goal", t)

# ---- Test case 1: all algorithms on one balanced scenario (Setup 1) ----
sc = Scenario(vehicle="car", time_slot="17:00-21:00", seed=42,
              gender="female", alone=True, pace="relaxed", weight_profile="balanced")
h = build_admissible_heuristic_cache(G, t, sc)
runners = [("BFS", bfs), ("UCS", ucs), ("Greedy Best-First", greedy_best_first),
           ("A*", astar), ("Weighted A*", weighted_astar)]
results = {}
print("\n=== TEST CASE 1 (car, evening peak, woman alone, balanced) ===")
print(f"{'algorithm':<18}{'found':>6}{'cost':>12}{'dist_m':>10}{'expanded':>10}{'runtime_ms':>12}")
routes, names, cols = [], [], []
palette = ["#377eb8", "#4daf4a", "#984ea3", "#e41a1c", "#ff7f00"]
for i, (nm, fn) in enumerate(runners):
    r = fn(G, s, t, sc, h_cache=h) if nm not in ("BFS", "UCS") else fn(G, s, t, sc)
    results[nm] = r
    print(f"{nm:<18}{str(r.found):>6}{r.total_cost:>12.0f}{r.distance_m:>10.0f}"
          f"{r.nodes_expanded:>10}{r.runtime_ms:>12.2f}")
    if r.found and len(r.path) >= 2:
        routes.append(r.path); names.append(nm); cols.append(palette[i])

# overlay map of all algorithms' routes, ZOOMED to the route bounding box
def _zoom_to_routes(ax, rts, margin=0.02):
    xs = [G.nodes[n]["x"] for r in rts for n in r]
    ys = [G.nodes[n]["y"] for r in rts for n in r]
    dx = (max(xs) - min(xs)) or 0.01; dy = (max(ys) - min(ys)) or 0.01
    ax.set_xlim(min(xs) - margin*dx*10 - 0.002, max(xs) + margin*dx*10 + 0.002)
    ax.set_ylim(min(ys) - margin*dy*10 - 0.002, max(ys) + margin*dy*10 + 0.002)

fig, ax = ox.plot_graph_routes(G, routes, route_colors=cols, route_linewidths=[3.2]*len(routes),
                               route_alpha=0.8, node_size=0, bgcolor="white",
                               edge_color="#d7dce3", edge_linewidth=0.7, orig_dest_size=60,
                               show=False, close=False)
_zoom_to_routes(ax, routes)
ax.legend(handles=[Line2D([0],[0],color=cols[i],lw=3,label=names[i]) for i in range(len(names))],
          loc="upper right", fontsize=8, framealpha=0.95)
ax.set_title("Test case: Uttara → Dhanmondi — optimal route on Dhaka roads", fontsize=10)
fig.savefig(FIG / "a1_routes_algorithms.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved a1_routes_algorithms.png")

# ---- Test case 2: A* for car vs bus -> different routes (bus barred from narrow roads) ----
print("\n=== TEST CASE 2 (A*, car vs bus; vehicle feasibility changes route) ===")
prof_routes, prof_names, prof_cols = [], [], []
for veh, col in [("car", "#e41a1c"), ("bus", "#377eb8")]:
    scp = Scenario(vehicle=veh, time_slot="17:00-21:00", seed=42,
                   gender="male", alone=False, pace="relaxed", weight_profile="balanced")
    hp = build_admissible_heuristic_cache(G, t, scp)
    r = astar(G, s, t, scp, h_cache=hp)
    print(f"{veh:<10} found={r.found} cost={r.total_cost:.0f}  dist_m={r.distance_m:.0f}  expanded={r.nodes_expanded}")
    if r.found:
        prof_routes.append(r.path); prof_names.append(f"A* ({veh})"); prof_cols.append(col)

if len(prof_routes) >= 1:
    fig, ax = ox.plot_graph_routes(G, prof_routes, route_colors=prof_cols,
                                   route_linewidths=[3]*len(prof_routes), route_alpha=0.8,
                                   node_size=0, bgcolor="white", edge_color="#d7dce3",
                                   edge_linewidth=0.5, show=False, close=False)
    ax.legend(handles=[Line2D([0],[0],color=prof_cols[i],lw=3,label=prof_names[i]) for i in range(len(prof_names))],
              loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_title("Same origin/destination, different preference profile → different route", fontsize=10)
    fig.savefig(FIG / "a1_routes_profiles.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved a1_routes_profiles.png")
print("\nDONE")
