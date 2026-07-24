"""
experiments.py — parameter / hyperparameter sweeps -> TSV
=========================================================
Runs the CCTV optimizer over many configurations and logs every run as one row
in a TSV file, so you can build the report's tables & plots.

Usage
-----
  python experiments.py                      # runs the 'all' suite
  python experiments.py --suite generations  # just one suite
  python experiments.py --suite budget area  # several suites
  python experiments.py --out results/my.tsv # custom output path

Suites
------
  baseline     greedy vs GA vs PSO on the default scenario
  generations  GA/PSO with generations/iterations = 20,50,100,200  (convergence)
  population   GA pop & PSO swarm = 20,40,80                        (pop size effect)
  budget       total cameras K = 5,10,15,20,30                      (more cameras)
  range        camera range r90/r120 scaled 0.7x,1.0x,1.3x          (sensor reach)
  area         dhanmondi, gulshan, mirpur, old_dhaka, uttara        (transferability)
  seed         GA/PSO over 5 seeds                                  (stochastic spread)
  hotspot      POI hotspot weighting on vs off                      (objective change)
"""
import argparse
import itertools
from pathlib import Path
import numpy as np

from cctv_world import build_world, AREAS
from data_loader import load_dhaka_graph
from solvers import SOLVERS

OUT_DEFAULT = Path(__file__).parent / "results" / "sweep.tsv"

COLUMNS = ["experiment", "area", "radius", "K", "n90", "n120", "r90", "r120",
           "hotspot", "crime", "overlap", "solver", "pop", "generations",
           "swarm", "iterations", "ants", "seed", "coverage_pct", "overlap_pct",
           "objective", "coverage_weight", "total_weight", "iters_run", "runtime"]

# baseline configuration every suite starts from
BASE = dict(area="dhanmondi", radius=1200, n90=5, n120=10, r90=200.0, r120=130.0,
            hotspot=1, crime=1, overlap=0.0, pop=40, generations=200, swarm=30,
            iterations=200, ants=20, patience=20, seed=42)

ALL_SOLVERS = ("greedy", "ga", "pso", "aco")


# ---------------------------------------------------------------------------
# suite definitions -> each yields a list of (experiment_name, config) rows
# ---------------------------------------------------------------------------
def _cfg(**over):
    c = dict(BASE); c.update(over); return c


def suite_baseline():
    for s in ALL_SOLVERS:
        yield "baseline", _cfg(solver=s)


def suite_generations():
    for n in (20, 50, 100, 200):
        yield "generations", _cfg(solver="ga", generations=n, patience=999)
        yield "generations", _cfg(solver="pso", iterations=n, patience=999)
        yield "generations", _cfg(solver="aco", iterations=n, patience=999)


def suite_population():
    for p in (20, 40, 80):
        yield "population", _cfg(solver="ga", pop=p)
        yield "population", _cfg(solver="pso", swarm=p)
        yield "population", _cfg(solver="aco", ants=p)


def suite_budget():
    # keep 1:2 ratio of 90:120 cameras
    for k in (5, 10, 15, 20, 30):
        n90, n120 = k // 3, k - k // 3
        for s in ALL_SOLVERS:
            yield "budget", _cfg(solver=s, n90=n90, n120=n120)


def suite_range():
    for f in (0.7, 1.0, 1.3):
        for s in ALL_SOLVERS:
            yield "range", _cfg(solver=s, r90=200.0 * f, r120=130.0 * f)


def suite_area():
    for ar in AREAS:
        for s in ALL_SOLVERS:
            yield "area", _cfg(solver=s, area=ar)


def suite_seed():
    for sd in (1, 7, 42, 100, 2024):
        for s in ("ga", "pso", "aco"):
            yield "seed", _cfg(solver=s, seed=sd)


def suite_hotspot():
    for h in (0, 1):
        for s in ALL_SOLVERS:
            yield "hotspot", _cfg(solver=s, hotspot=h)


def suite_crime():
    for cr in (0, 1):
        for s in ALL_SOLVERS:
            yield "crime", _cfg(solver=s, crime=cr)


def suite_overlap():
    # DENSE scenario (small area, many long-range cameras) so overlap matters;
    # this is where the objective becomes non-submodular and PSO can beat greedy.
    dense = dict(radius=700, n90=12, n120=18, r90=320.0, r120=240.0,
                 generations=300, iterations=300, ants=30, patience=40)
    for lam in (0.0, 0.5, 1.0, 2.0):
        for s in ALL_SOLVERS:
            yield "overlap", _cfg(solver=s, overlap=lam, **dense)


def suite_greedy_fail():
    """Extreme dense scenario where greedy provably fails.

    Setup: tiny area (r=500m), 30 cameras with very long range (r=450m) --
    almost the entire area is covered by a single camera.  With high overlap
    penalty (lambda >= 5) greedy clusters cameras at the highest-demand hub
    and destroys the objective, while PSO spreads them out and wins clearly.
    """
    extreme = dict(
        radius=500,                          # tiny, packed area
        n90=15, n120=15,                     # 30 cameras total
        r90=450.0, r120=450.0,              # huge range: 1 camera ~ covers 40%+
        generations=500, iterations=500,
        ants=40, patience=100,
        crime=1, hotspot=1,
    )
    for lam in (0.0, 5.0, 10.0, 20.0):
        for s in ("greedy", "pso"):          # only greedy vs PSO for clarity
            yield "greedy_fail", _cfg(solver=s, overlap=lam, **extreme)


SUITES = {
    "baseline": suite_baseline, "generations": suite_generations,
    "population": suite_population, "budget": suite_budget,
    "range": suite_range, "area": suite_area, "seed": suite_seed,
    "hotspot": suite_hotspot, "crime": suite_crime, "overlap": suite_overlap,
    "greedy_fail": suite_greedy_fail,
}


# ---------------------------------------------------------------------------
def build_specs(c):
    hf90, hf120 = np.deg2rad(45), np.deg2rad(60)
    return [(c["r90"], hf90)] * c["n90"] + [(c["r120"], hf120)] * c["n120"]


def run_all(suite_names, out_path):
    G = load_dhaka_graph()                 # load the big graph ONCE
    world_cache = {}                        # reuse worlds across configs

    configs = []
    for name in suite_names:
        configs.extend(list(SUITES[name]()))
    print(f"[exp] {len(configs)} runs across suites: {', '.join(suite_names)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")

        for i, (exp, c) in enumerate(configs, 1):
            specs = build_specs(c)
            K = len(specs)
            max_R = max(c["r90"], c["r120"])
            wkey = (c["area"], c["radius"], round(max_R, 1), c["hotspot"],
                    c["crime"])
            if wkey not in world_cache:
                world_cache[wkey] = build_world(
                    area=c["area"], radius_m=c["radius"], max_R=max_R,
                    hotspot=bool(c["hotspot"]), crime=bool(c["crime"]),
                    seed=c["seed"], verbose=False, G=G)
            world = world_cache[wkey]
            world.overlap_penalty = c["overlap"]        # scalar, set per config

            # greedy warm-start for GA/PSO/ACO (same as main.py, fair comparison)
            kw = dict(seed=c["seed"], generations=c["generations"],
                      pop_size=c["pop"], iterations=c["iterations"],
                      swarm=c["swarm"], ants=c["ants"], patience=c["patience"],
                      verbose=False)
            if c["solver"] == "greedy":
                r = SOLVERS["greedy"](world, specs, seed=c["seed"])
            else:
                seed_g = SOLVERS["greedy"](world, specs, seed=c["seed"])["genome"]
                r = SOLVERS[c["solver"]](world, specs, seed_genome=seed_g, **kw)

            # compute objective = coverage - lambda * overlap (actual score optimized)
            obj = r['coverage_pct'] - c['overlap'] * r['overlap_pct']

            row = [exp, c["area"], c["radius"], K, c["n90"], c["n120"],
                   f"{c['r90']:.0f}", f"{c['r120']:.0f}", c["hotspot"],
                   c["crime"], f"{c['overlap']:.1f}", c["solver"], c["pop"],
                   c["generations"], c["swarm"], c["iterations"], c["ants"],
                   c["seed"], f"{r['coverage_pct']:.2f}",
                   f"{r['overlap_pct']:.2f}", f"{obj:.2f}",
                   f"{r['coverage_weight']:.0f}",
                   f"{world.total_weight:.0f}", r.get("iters") or "-",
                   f"{r['runtime']:.3f}"]
            f.write("\t".join(map(str, row)) + "\n")
            f.flush()
            print(f"  [{i:>3}/{len(configs)}] {exp:<11} {c['solver']:<6} "
                  f"K={K:<2} gen={c['generations']:<3} -> "
                  f"{r['coverage_pct']:5.1f}%  {r['runtime']:.2f}s")

    print(f"\n[done] TSV saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", nargs="+", default=["all"],
                   help="one or more of: " + ", ".join(SUITES) + ", all")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    a = p.parse_args()
    names = list(SUITES) if "all" in a.suite else a.suite
    run_all(names, Path(a.out))


if __name__ == "__main__":
    main()
