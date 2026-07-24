
import argparse
from pathlib import Path

from delivery_env import DeliveryEnv, AREAS
from data_loader import load_dhaka_graph
from vi_agent import ValueIterationAgent
from ql_agent import QLearningAgent

OUT_DEFAULT = Path(__file__).parent / "results" / "sweep.tsv"

COLUMNS = ["experiment", "area", "nodes", "actual_nodes", "states", "targets",
           "urgent", "jam", "flood", "gamma", "agent", "episodes", "alpha",
           "epsilon", "seed", "route_reward", "iters_run", "runtime"]

BASE = dict(area="dhanmondi", nodes=150, targets=3, urgent=1, jam=0.18,
            flood=0.12, gamma=0.95, epsilon=1.0, episodes=3000, alpha=0.1,
            seed=42)

ALL = ("vi", "ql")


def _cfg(**over):
    c = dict(BASE); c.update(over); return c


def suite_baseline():
    for ag in ALL:
        yield "baseline", _cfg(agent=ag)


def suite_gamma():
    for g in (0.5, 0.8, 0.9, 0.95, 0.99):
        for ag in ALL:
            yield "gamma", _cfg(agent=ag, gamma=g)


def suite_targets():
    for t in (2, 3, 4):
        for ag in ALL:
            yield "targets", _cfg(agent=ag, targets=t, urgent=min(1, t))


def suite_nodes():
    for n in (60, 100, 150, 200):
        for ag in ALL:
            yield "nodes", _cfg(agent=ag, nodes=n)


def suite_urgent():
    for u in (0, 1, 2):
        for ag in ALL:
            yield "urgent", _cfg(agent=ag, urgent=u)


def suite_hazards():
    for jam, flood in ((0.05, 0.05), (0.18, 0.12), (0.35, 0.30)):
        for ag in ALL:
            yield "hazards", _cfg(agent=ag, jam=jam, flood=flood)


def suite_episodes():
    for e in (500, 1000, 2000, 3000, 5000):
        yield "episodes", _cfg(agent="ql", episodes=e)


def suite_alpha():
    for al in (0.05, 0.1, 0.3):
        yield "alpha", _cfg(agent="ql", alpha=al)


def suite_seed():
    for s in (1, 7, 42, 100, 2024):
        for ag in ALL:
            yield "seed", _cfg(agent=ag, seed=s)


SUITES = {
    "baseline": suite_baseline, "gamma": suite_gamma, "targets": suite_targets,
    "nodes": suite_nodes, "urgent": suite_urgent, "hazards": suite_hazards,
    "episodes": suite_episodes, "alpha": suite_alpha, "seed": suite_seed,
}


def run_all(suite_names, out_path):
    G = load_dhaka_graph()
    configs = []
    for name in suite_names:
        configs.extend(list(SUITES[name]()))
    print(f"[exp] {len(configs)} runs across: {', '.join(suite_names)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for i, (exp, c) in enumerate(configs, 1):
            env = DeliveryEnv(area=c["area"], n_nodes=c["nodes"],
                              n_targets=c["targets"], n_urgent=c["urgent"],
                              jam_frac=c["jam"], flood_frac=c["flood"],
                              gamma=c["gamma"], seed=c["seed"], G=G, verbose=False)
            an = env.graph.number_of_nodes()
            states = an * (2 ** c["targets"])

            if c["agent"] == "vi":
                ag = ValueIterationAgent(env, epsilon=c["epsilon"], verbose=False)
                pi = ag.solve()
                iters = ag.iters
            else:
                ag = QLearningAgent(env, alpha=c["alpha"], episodes=c["episodes"],
                                    seed=c["seed"], verbose=False)
                pi = ag.train()
                iters = ag.iters   # actual episodes run (may be < episodes if converged early)
            reward = env.evaluate_policy(pi)

            row = [exp, c["area"], c["nodes"], an, states, c["targets"],
                   c["urgent"], f"{c['jam']:.2f}", f"{c['flood']:.2f}",
                   f"{c['gamma']:.2f}", c["agent"], c["episodes"],
                   f"{c['alpha']:.2f}", f"{c['epsilon']:.1f}", c["seed"],
                   f"{reward:.0f}", iters, f"{ag.runtime:.3f}"]
            f.write("\t".join(map(str, row)) + "\n")
            f.flush()
            print(f"  [{i:>3}/{len(configs)}] {exp:<9} {c['agent']:<3} "
                  f"t={c['targets']} n={an} g={c['gamma']} -> "
                  f"reward={reward:.0f}  {ag.runtime:.2f}s")
    print(f"\n[done] TSV saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", nargs="+", default=["all"])
    p.add_argument("--out", default=str(OUT_DEFAULT))
    a = p.parse_args()
    names = list(SUITES) if "all" in a.suite else a.suite
    run_all(names, Path(a.out))


if __name__ == "__main__":
    main()
