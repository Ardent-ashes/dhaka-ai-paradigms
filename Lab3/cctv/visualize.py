"""
visualize.py
============
Renders the CCTV-placement results:

  * coverage map  -> roads, demand points (green=covered / grey=not),
                     cameras as markers with their field-of-view WEDGES.
  * convergence   -> best coverage vs generation/iteration (GA & PSO).
  * comparison    -> coverage% and runtime across solvers.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from shapely.geometry import LineString

RAD2DEG = 180.0 / np.pi


def _plot_roads(ax, world, color="0.82", lw=0.7):
    Hp = world.Hp
    seen = set()
    for u, v, data in Hp.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        if "geometry" in data and isinstance(data["geometry"], LineString):
            xs, ys = data["geometry"].xy
        else:
            xs = [Hp.nodes[u]["x"], Hp.nodes[v]["x"]]
            ys = [Hp.nodes[u]["y"], Hp.nodes[v]["y"]]
        ax.plot(xs, ys, color=color, lw=lw, zorder=1)


# distinct colours per camera FOV type (90 deg, 120 deg, ...)
_TYPE_COLORS = ["#b2182b", "#762a83", "#1b7837", "#e08214", "#2166ac"]


def _fov_deg(hf):
    return int(round(2 * hf * RAD2DEG))


def plot_map(world, result, path, title=None, area=None):
    cams = result["cameras"]
    covered = world.covered_mask(cams)
    fig, ax = plt.subplots(figsize=(12, 12))
    _plot_roads(ax, world)

    # demand points
    ax.scatter(world.dx[~covered], world.dy[~covered], s=3, c="0.7",
               label="road / junction (not covered)", zorder=2)
    ax.scatter(world.dx[covered], world.dy[covered], s=6, c="#1a9850",
               label="covered", zorder=3)

    # map each FOV type -> a colour
    fovs = sorted({_fov_deg(hf) for *_, hf in cams})
    color_of = {f: _TYPE_COLORS[i % len(_TYPE_COLORS)] for i, f in enumerate(fovs)}

    # field-of-view wedges
    for c, th, R, hf in cams:
        cx, cy = world.cand_xy[c]
        col = color_of[_fov_deg(hf)]
        ax.add_patch(Wedge((cx, cy), R, (th - hf) * RAD2DEG, (th + hf) * RAD2DEG,
                           facecolor=col, alpha=0.15, edgecolor=col,
                           lw=0.6, zorder=4))

    # camera markers, coloured by type + numbered
    for i, (c, th, R, hf) in enumerate(cams, start=1):
        cx, cy = world.cand_xy[c]
        col = color_of[_fov_deg(hf)]
        ax.scatter([cx], [cy], marker="^", s=110, c=col, edgecolors="k",
                   linewidths=0.7, zorder=5)
        ax.annotate(str(i), (cx, cy), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, fontweight="bold",
                    color="k", zorder=6)

    # legend: one entry per camera type
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="^", color="w", markerfacecolor=color_of[f],
                      markeredgecolor="k", markersize=11,
                      label=f"{f}° camera") for f in fovs]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a9850",
                       markersize=8, label="covered"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7",
                       markersize=8, label="not covered")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, fontsize=10)

    # scale bar (500 m) -- coords are in metres
    xmin, xmax, ymin, ymax = world.bbox
    x0, y0 = xmin + 40, ymin - 20
    ax.plot([x0, x0 + 500], [y0, y0], "k-", lw=3, zorder=7)
    ax.annotate("500 m", (x0 + 250, y0), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9)

    ax.set_aspect("equal")
    head = f"{area or ''}  |  " if area else ""
    ax.set_title(title or f"{head}{result['solver'].upper()} — "
                 f"coverage {result['coverage_pct']:.1f}%  "
                 f"({len(cams)} cameras)", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_crime(world, result, path, area=None):
    """Synthetic crime/crowd heatmap with the chosen cameras on top -- shows
    that cameras gravitate toward high-risk zones."""
    if world.crime is None:
        return None
    fig, ax = plt.subplots(figsize=(12, 12))
    _plot_roads(ax, world, color="0.88", lw=0.6)
    sc = ax.scatter(world.dx, world.dy, c=world.crime, s=8, cmap="YlOrRd",
                    zorder=2)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="synthetic crime / crowd score")
    cams = result["cameras"]
    ax.scatter(world.cand_xy[[c for c, *_ in cams], 0],
               world.cand_xy[[c for c, *_ in cams], 1],
               marker="^", s=120, c="#08306b", edgecolors="w",
               linewidths=0.8, zorder=5, label="camera")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title(f"{(area or '')}  |  Crime/crowd risk vs camera placement "
                 f"({result['solver'].upper()})", fontsize=13)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_convergence(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for r in results:
        if r.get("history"):
            total = r["_total_weight"]
            ax.plot(100 * np.array(r["history"]) / total,
                    label=r["solver"].upper(), lw=2)
            plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel("generation / iteration")
    ax.set_ylabel("best coverage (%)")
    ax.set_title("Convergence")
    ax.legend(); ax.grid(alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_comparison(results, path):
    names = [r["solver"].upper() for r in results]
    cov = [r["coverage_pct"] for r in results]
    rt = [r["runtime"] for r in results]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(names, cov, color="#1a9850")
    a1.set_ylabel("coverage (%)"); a1.set_title("Coverage")
    for i, v in enumerate(cov):
        a1.text(i, v + 0.5, f"{v:.1f}", ha="center")
    a2.bar(names, rt, color="#2166ac")
    a2.set_ylabel("runtime (s)"); a2.set_title("Runtime")
    for i, v in enumerate(rt):
        a2.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
