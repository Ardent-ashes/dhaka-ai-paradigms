"""
main.py — CCTV Coverage Optimizer (CLI)
=======================================
Place K fixed cameras on real Dhaka OSM roads to MAXIMISE weighted coverage
(road importance + intersections + POI hotspots), then generate a map.

Interactive:   python main.py
One-shot:      python main.py --auto
Override:      python main.py --auto --area gulshan --n90 8 --n120 12
"""
import argparse
from pathlib import Path
import numpy as np

from cctv_world import build_world, AREAS
from solvers import SOLVERS
import visualize as viz

OUT = Path(__file__).parent / "output"


def ask(prompt, default, cast=str):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        print("  invalid, using default.")
        return default


def parse_args():
    p = argparse.ArgumentParser(description="CCTV Coverage Optimizer")
    p.add_argument("--auto", action="store_true", help="skip prompts, use values below")
    p.add_argument("--area", default="dhanmondi", choices=list(AREAS))
    p.add_argument("--radius", type=int, default=1200)
    p.add_argument("--n90", type=int, default=5, help="# of 90-degree cameras")
    p.add_argument("--n120", type=int, default=10, help="# of 120-degree cameras")
    p.add_argument("--r90", type=float, default=200.0, help="range of 90-deg cam (m)")
    p.add_argument("--r120", type=float, default=130.0, help="range of 120-deg cam (m)")
    p.add_argument("--hotspot", type=int, default=1, help="1=use OSM POI hotspots")
    p.add_argument("--crime", type=int, default=1, help="1=synthetic crime layer")
    p.add_argument("--overlap", type=float, default=0.0,
                   help="overlap penalty lambda (0=off; try 1.0 to punish redundancy)")
    p.add_argument("--solver", default="all",
                   choices=["ga", "pso", "aco", "all"],
                   help="solver to run (greedy used internally as warm-start only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--swarm", type=int, default=30)
    p.add_argument("--ants", type=int, default=20)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def gather(a):
    """Interactively fill in values (unless --auto)."""
    if a.auto:
        return a
    print("\n=== CCTV Coverage Optimizer — setup (Enter = default) ===")
    a.area = ask(f"Area {list(AREAS)}", a.area)
    a.radius = ask("Study radius (m)", a.radius, int)
    print("\nCamera fleet (K fixed = n90 + n120):")
    a.n90 = ask("  # of 90-degree cameras", a.n90, int)
    a.n120 = ask("  # of 120-degree cameras", a.n120, int)
    a.r90 = ask("  range of 90-degree camera (m)", a.r90, float)
    a.r120 = ask("  range of 120-degree camera (m)", a.r120, float)
    a.hotspot = ask("Use OSM POI hotspots? 1/0", a.hotspot, int)
    a.crime = ask("Use synthetic crime layer? 1/0", a.crime, int)
    a.overlap = ask("Overlap penalty lambda (0=off)", a.overlap, float)
    a.solver = ask("Solver [ga/pso/aco/all]", a.solver)
    a.seed = ask("Random seed", a.seed, int)
    return a


def build_specs(a):
    hf90 = np.deg2rad(90 / 2)
    hf120 = np.deg2rad(120 / 2)
    specs = [(a.r90, hf90)] * a.n90 + [(a.r120, hf120)] * a.n120
    return specs


def main():
    a = gather(parse_args())
    specs = build_specs(a)
    K = len(specs)
    if K == 0:
        print("No cameras requested."); return
    max_R = max(a.r90, a.r120)

    print(f"\n[run] area={a.area} K={K} (90°×{a.n90}, 120°×{a.n120}) "
          f"R=({a.r90:.0f}/{a.r120:.0f})m solver={a.solver}\n")

    world = build_world(area=a.area, radius_m=a.radius, max_R=max_R,
                        hotspot=bool(a.hotspot), crime=bool(a.crime),
                        overlap_penalty=a.overlap, seed=a.seed)

    names = ["greedy", "ga", "pso", "aco"] if a.solver == "all" else [a.solver]
    kw = dict(seed=a.seed, generations=a.generations, pop_size=a.pop,
              iterations=a.iterations, swarm=a.swarm, ants=a.ants,
              patience=a.patience)

    # Greedy runs internally as warm-start for GA/PSO/ACO (not shown as a solver).
    print("[seed] greedy warm-start ...", flush=True)
    seed_res = SOLVERS["greedy"](world, specs, seed=a.seed)
    seed_genome = seed_res["genome"]

    results = []
    names = ["ga", "pso", "aco"] if a.solver == "all" else [a.solver]
    for name in names:
        print(f"[solve] {name} ...", flush=True)
        r = SOLVERS[name](world, specs, seed_genome=seed_genome, **kw)
        r["_total_weight"] = world.total_weight
        results.append(r)
        print(f"        coverage {r['coverage_pct']:.1f}%  "
              f"in {r['runtime']:.2f}s")

    run_id = f"{a.area}_K{K}_{a.solver}"
    outdir = OUT / run_id
    print(f"\n[out] writing to {outdir}")
    for r in results:
        viz.plot_map(world, r, outdir / f"map_{r['solver']}.png",
                     area=a.area.title())
    viz.plot_convergence(results, outdir / "convergence.png")
    if len(results) > 1:
        viz.plot_comparison(results, outdir / "comparison.png")
    if world.crime is not None:
        viz.plot_crime(world, results[-1], outdir / "crime_heatmap.png",
                       area=a.area.title())

    print("\n=== SUMMARY ===")
    print(f"{'solver':<8}{'coverage%':>11}{'overlap%':>10}{'runtime(s)':>12}{'iters':>8}")
    for r in results:
        print(f"{r['solver']:<8}{r['coverage_pct']:>11.1f}{r['overlap_pct']:>10.1f}"
              f"{r['runtime']:>12.2f}{(r.get('iters') or '-'):>8}")
    print(f"\nMaps + charts saved in: {outdir}")


if __name__ == "__main__":
    main()
