
import argparse
from pathlib import Path

from delivery_env import DeliveryEnv, AREAS
from vi_agent import ValueIterationAgent
from ql_agent import QLearningAgent
import visualize as viz

OUT = Path(__file__).parent / "output"


def ask(prompt, default, cast=str):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        print("  invalid, using default."); return default


def parse_args():
    p = argparse.ArgumentParser(description="TiffinTime delivery MDP")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--show-map", action="store_true",
                   help="save a numbered node map and exit (use to pick depot/targets)")
    p.add_argument("--area", default="dhanmondi", choices=list(AREAS))
    p.add_argument("--nodes", type=int, default=150)
    p.add_argument("--targets", type=int, default=3)
    p.add_argument("--urgent", type=int, default=1)
    p.add_argument("--depot", type=int, default=None,
                   help="node index for depot (see --show-map). Default: random")
    p.add_argument("--node-targets", type=int, nargs="+", default=None,
                   help="node indices for delivery stops (see --show-map). Default: random")
    p.add_argument("--jam", type=float, default=0.18)
    p.add_argument("--flood", type=float, default=0.12)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=1.0, help="VI stop threshold")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agent", default="both", choices=["vi", "ql", "both"])
    return p.parse_args()


def gather(a):
    if a.auto or a.show_map:
        return a
    print("\n=== TiffinTime delivery MDP — setup (Enter = default) ===")
    a.area = ask(f"Area {list(AREAS)}", a.area)
    a.nodes = ask("Sub-graph size (nodes)", a.nodes, int)

    use_custom = ask("Specify depot/targets manually? (y/n)", "n")
    if use_custom.lower() == "y":
        print("  → First run with --show-map to see node indices on the map.")
        a.depot = ask("  Depot node index", 0, int)
        raw = input(f"  Target node indices (space-separated) [random]: ").strip()
        a.node_targets = list(map(int, raw.split())) if raw else None
        if a.node_targets:
            a.targets = len(a.node_targets)
    else:
        a.depot = None
        a.node_targets = None
        a.targets = ask("Number of delivery stops (keep 2-4!)", a.targets, int)

    a.urgent = ask("How many of them are URGENT", a.urgent, int)
    a.agent = ask("Agent [vi/ql/both]", a.agent)
    a.seed = ask("Random seed", a.seed, int)
    return a


def rollout_reward(env, policy, max_steps=300):
    """Deterministic route reward (follow policy assuming intended moves)."""
    s = env.reset(); total = 0.0; seen = set()
    for _ in range(max_steps):
        if env.is_terminal(s) or s in seen:
            break
        seen.add(s)
        a = policy.get(s)
        if a is None:
            break
        succ = [(p, sp, r) for p, sp, r in env.transitions(s, a) if sp[0] == a]
        if not succ:
            break
        _, sp, r = succ[0]
        total += r; s = sp
    return total


def delivery_order(env, route):
    tset = set(env.targets)
    return [n for n in route if n in tset]


def main():
    a = gather(parse_args())

    # ── show-map mode: generate node index map and exit ──────────────────────
    if a.show_map:
        print(f"\n[map] Building env for {a.area} (nodes={a.nodes}, seed={a.seed}) ...")
        env = DeliveryEnv(area=a.area, n_nodes=a.nodes, n_targets=a.targets,
                          n_urgent=a.urgent, jam_frac=a.jam, flood_frac=a.flood,
                          gamma=a.gamma, seed=a.seed)
        map_path = OUT / f"{a.area}_n{a.nodes}_node_map.png"
        viz.plot_node_map(env, map_path)
        print(f"\n[map] Total nodes: {len(env.nodes)}  (valid indices: 0 – {len(env.nodes)-1})")
        print(f"[map] Example usage:")
        print(f"  python main.py --area {a.area} --nodes {a.nodes} "
              f"--depot 5 --node-targets 12 34 56")
        return

    # ── normal run ────────────────────────────────────────────────────────────
    node_targets = getattr(a, "node_targets", None)
    node_depot   = getattr(a, "depot", None)
    if node_targets:
        a.targets = len(node_targets)
        a.urgent  = min(a.urgent, a.targets)

    if a.targets > 6:
        print("[warn] >6 targets -> 2^k state blow-up; Value Iteration may be slow.")

    print(f"\n[run] area={a.area} nodes={a.nodes} targets={a.targets} "
          f"urgent={a.urgent} gamma={a.gamma} agent={a.agent}")
    if node_depot is not None:
        print(f"[run] custom depot=#{node_depot}, targets=#{node_targets}\n")
    else:
        print(f"[run] depot/targets: random (seed={a.seed})\n")

    env = DeliveryEnv(area=a.area, n_nodes=a.nodes, n_targets=a.targets,
                      n_urgent=a.urgent, jam_frac=a.jam, flood_frac=a.flood,
                      gamma=a.gamma, seed=a.seed,
                      node_depot=node_depot, node_targets=node_targets)
    n_states = env.graph.number_of_nodes() * (2 ** a.targets)
    print(f"[env] ~{n_states} states ({env.graph.number_of_nodes()} nodes x "
          f"2^{a.targets})\n")

    outdir = OUT / f"{a.area}_n{a.nodes}_t{a.targets}"
    results = []

    if a.agent in ("vi", "both"):
        print("[solve] Value Iteration ...")
        vi = ValueIterationAgent(env, epsilon=a.epsilon)
        pi = vi.solve()
        route = env.greedy_route(pi)
        rew = rollout_reward(env, pi)
        viz.plot_vi_convergence(vi.history, outdir / "vi_convergence.png")
        viz.plot_vi_curve(vi.rewards, outdir / "vi_learning_curve.png")
        viz.plot_route(env, route, outdir / "route_vi.png",
                       title=f"Value Iteration route — reward {rew:.0f}")
        print(f"        route reward {rew:.0f}, "
              f"delivery order {delivery_order(env, route)}")
        results.append({"solver": "VI", "reward": rew, "runtime": vi.runtime})

    if a.agent in ("ql", "both"):
        print("[solve] Q-Learning ...")
        ql = QLearningAgent(env, alpha=a.alpha, episodes=a.episodes, seed=a.seed)
        pi = ql.train()
        route = env.greedy_route(pi)
        rew = rollout_reward(env, pi)
        viz.plot_ql_curve(ql.rewards, outdir / "ql_learning_curve.png",
                          conv_ep=ql.conv_ep)
        viz.plot_route(env, route, outdir / "route_ql.png",
                       title=f"Q-Learning route — reward {rew:.0f}")
        print(f"        route reward {rew:.0f}, "
              f"delivery order {delivery_order(env, route)}")
        results.append({"solver": "QL", "reward": rew, "runtime": ql.runtime,
                        "iters": ql.iters})

    if len(results) > 1:
        viz.plot_comparison(results, outdir / "comparison.png")
        # combined learning curve (VI reward/iter vs QL reward/episode)
        viz.plot_learning_curves(
            vi.rewards, ql.rewards,
            outdir / "learning_curves_combined.png",
            ql_conv_ep=ql.conv_ep
        )

    print("\n=== SUMMARY ===")
    print(f"{'agent':<6}{'route_reward':>14}{'runtime(s)':>12}")
    for r in results:
        print(f"{r['solver']:<6}{r['reward']:>14.0f}{r['runtime']:>12.2f}")
    print(f"\nPlots saved in: {outdir}")


if __name__ == "__main__":
    main()
