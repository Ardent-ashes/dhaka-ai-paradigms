"""
visualize.py
============
Plots for the delivery MDP:
  * Value-Iteration convergence  (delta vs sweep)
  * Q-Learning learning curve     (episode vs total reward, smoothed)
  * a comparison of the two
  * the final delivery ROUTE on the real Dhaka map (hazards highlighted)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_node_map(env, path):
    """Save a numbered node map so the user can pick depot/target indices."""
    G = env.graph
    fig, ax = plt.subplots(figsize=(14, 14))

    # all roads (light grey)
    for u, v in env.length.keys():
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="#d4d4d4", lw=0.7, zorder=1)

    # hazard roads
    for (u, v), h in env.hazard.items():
        if h == "normal":
            continue
        col = "#f46d43" if h == "jam" else "#4575b4"
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color=col, lw=1.5, alpha=0.4, zorder=2)

    # nodes with sequential index labels
    for idx, node in enumerate(env.nodes):
        x, y = G.nodes[node]["x"], G.nodes[node]["y"]
        ax.scatter(x, y, s=45, c="#2b83ba", edgecolors="#1a5276",
                   linewidths=0.5, zorder=5)
        ax.annotate(str(idx), (x, y),
                    fontsize=5.5, ha="center", va="bottom", fontweight="bold",
                    color="#1a1a1a",
                    xytext=(0, 5), textcoords="offset points", zorder=6)

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        "Node Index Map  —  use index numbers with:  "
        "python main.py --depot <N> --node-targets <N> <N> ...",
        fontsize=10, pad=12)

    # legend for hazards
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#f46d43", lw=2, label="jam road"),
        Line2D([0], [0], color="#4575b4", lw=2, label="flooded road"),
    ]
    ax.legend(handles=legend, loc="upper right", framealpha=0.9)
    _save(fig, path)
    print(f"[map] Node index map saved → {path}")


def plot_vi_convergence(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color="#2166ac", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration (sweep)")
    ax.set_ylabel("max utility change (delta, log)")
    ax.set_title("Value Iteration — convergence (delta)")
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_vi_curve(rewards, path):
    """VI reward per iteration — same style as QL learning curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = range(1, len(rewards) + 1)
    ax.plot(iters, rewards, color="#bbbbbb", lw=0.8, label="per iteration")
    # smooth line (simple: just connect since VI has few iterations)
    ax.plot(iters, rewards, color="#2166ac", lw=2.5, marker="o",
            markersize=4, label="greedy policy reward")
    ax.set_xlabel("iteration (sweep)")
    ax.set_ylabel("total reward (greedy rollout)")
    ax.set_title("Value Iteration — learning curve")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path)


def plot_learning_curves(vi_rewards, ql_rewards, path, ql_conv_ep=None, window=100):
    """Side-by-side VI and QL reward curves — same axes style for easy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # VI (left)
    iters = range(1, len(vi_rewards) + 1)
    ax1.plot(iters, vi_rewards, color="#bbbbbb", lw=0.8)
    ax1.plot(iters, vi_rewards, color="#2166ac", lw=2.5, marker="o", markersize=4)
    ax1.set_xlabel("iteration (sweep)"); ax1.set_ylabel("total reward (greedy rollout)")
    ax1.set_title("Value Iteration — reward per iteration")
    ax1.grid(alpha=0.3)

    # QL (right)
    ax2.plot(ql_rewards, color="#bbbbbb", lw=0.6, label="per episode")
    if len(ql_rewards) >= window:
        sm = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(ql_rewards)), sm,
                 color="#d7191c", lw=2, label=f"{window}-ep moving avg")
    if ql_conv_ep is not None:
        ax2.axvline(x=ql_conv_ep, color="#1a9641", lw=1.8, ls="--",
                    label=f"converged @ ep {ql_conv_ep}")
    ax2.set_xlabel("episode"); ax2.set_ylabel("total reward")
    ax2.set_title("Q-Learning — reward per episode")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Learning Curves: VI vs Q-Learning", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, path)


def plot_ql_curve(rewards, path, window=100, conv_ep=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rewards, color="#bbbbbb", lw=0.6, label="per episode")
    if len(rewards) >= window:
        sm = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), sm, color="#d7191c", lw=2,
                label=f"{window}-ep moving avg")
    # mark convergence point if early-stopped
    if conv_ep is not None:
        ax.axvline(x=conv_ep, color="#1a9641", lw=1.8, ls="--",
                   label=f"converged @ ep {conv_ep}")
        ax.annotate(f"converged\n@ ep {conv_ep}",
                    xy=(conv_ep, ax.get_ylim()[0]),
                    xytext=(conv_ep + max(10, len(rewards) * 0.03),
                            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1),
                    fontsize=8, color="#1a9641",
                    arrowprops=dict(arrowstyle="->", color="#1a9641"))
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.set_title("Q-Learning — learning curve")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path)


def plot_comparison(results, path):
    """results: list of dicts with keys solver, reward, runtime."""
    names = [r["solver"] for r in results]
    rew = [r["reward"] for r in results]
    rt = [r["runtime"] for r in results]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(names, rew, color="#1a9641")
    a1.set_ylabel("route reward (greedy rollout)"); a1.set_title("Policy quality")
    for i, v in enumerate(rew):
        a1.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    a2.bar(names, rt, color="#2166ac")
    a2.set_ylabel("runtime (s)"); a2.set_title("Compute time")
    for i, v in enumerate(rt):
        a2.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    _save(fig, path)


def plot_route(env, route, path, title="Delivery route"):
    G = env.graph
    fig, ax = plt.subplots(figsize=(11, 11))

    # all roads (light)
    for u, v in env.length.keys():
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="#e3e3e3", lw=0.7, zorder=1)

    # hazards
    seen_lbl = set()
    for (u, v), h in env.hazard.items():
        if h == "normal":
            continue
        col = "#f46d43" if h == "jam" else "#4575b4"
        lbl = "jam road" if h == "jam" else "flooded road"
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]], color=col, lw=2.2,
                alpha=0.6, zorder=2, label=None if lbl in seen_lbl else lbl)
        seen_lbl.add(lbl)

    # the route
    rx = [G.nodes[n]["x"] for n in route]
    ry = [G.nodes[n]["y"] for n in route]
    ax.plot(rx, ry, color="#d7191c", lw=3, zorder=4, label="delivery route")

    # depot + delivery stops
    ax.scatter([G.nodes[env.start]["x"]], [G.nodes[env.start]["y"]], s=200,
               c="#1a9641", marker="o", edgecolors="k", zorder=6, label="depot (start)")
    for t in env.targets:
        urgent = t in env.urgent
        ax.scatter([G.nodes[t]["x"]], [G.nodes[t]["y"]], s=260 if urgent else 180,
                   c="#fdae61" if urgent else "#2b83ba",
                   marker="*" if urgent else "s", edgecolors="k", zorder=7,
                   label="urgent stop" if urgent else "delivery stop")

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", framealpha=0.95)
    _save(fig, path)
