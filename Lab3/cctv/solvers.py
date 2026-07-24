import time
import numpy as np

TWO_PI = 2 * np.pi


def _cameras(world, genome, specs):
    """genome: list of (cand_index, theta) -> cameras for World helpers."""
    return [(c, th, specs[i][0], specs[i][1]) for i, (c, th) in enumerate(genome)]


def evaluate(world, genome, specs):
    # solvers MAXIMISE the objective (coverage minus overlap penalty);
    # with overlap_penalty=0 this is plain coverage.
    return world.objective(_cameras(world, genome, specs))


# Greedy baseline
def greedy(world, specs, n_dirs=16, seed=42, **_):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    dirs = np.linspace(0, TWO_PI, n_dirs, endpoint=False)
    count = np.zeros(world.dw.size, dtype=int)   # times each point is seen
    lam = world.overlap_penalty
    used = set()
    genome = []
    for R, hf in specs:
        best_delta, best_c, best_th = -1e18, None, 0.0
        for c in range(len(world.cand_ids)):
            if c in used:
                continue
            for th in dirs:
                idx = world.camera_cover_idx(c, th, R, hf)
                if idx.size == 0:
                    continue
                seen = count[idx]
                # objective change from adding this camera:
                new_cov = world.dw[idx[seen == 0]].sum()      # newly covered
                new_ovl = world.dw[idx[seen >= 1]].sum()      # becomes redundant
                delta = new_cov - lam * new_ovl
                if delta > best_delta:
                    best_delta, best_c, best_th = delta, c, th
        if best_c is None:                       # nothing left to cover
            best_c = int(rng.integers(len(world.cand_ids)))
        used.add(best_c)
        count[world.camera_cover_idx(best_c, best_th, R, hf)] += 1
        genome.append((best_c, best_th))
    return _result("greedy", world, genome, specs, t0)


# 2. Genetic Algorithm
def _rand_genome(world, K, rng):
    nodes = rng.choice(len(world.cand_ids), size=K, replace=False)
    thetas = rng.uniform(0, TWO_PI, size=K)
    return list(zip(nodes.tolist(), thetas.tolist()))


def _repair(genome, C, rng):
    """Ensure at most one camera per candidate node."""
    seen, out = set(), []
    for c, th in genome:
        while c in seen:
            c = int(rng.integers(C))
        seen.add(c); out.append((c, th))
    return out


def ga(world, specs, pop_size=40, generations=200, mutation_rate=0.2,
       crossover_rate=0.85, tournament=3, elitism=2, seed=42,
       seed_genome=None, patience=20, min_delta=1e-6, verbose=True, **_):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    K, C = len(specs), len(world.cand_ids)
    pop = [_rand_genome(world, K, rng) for _ in range(pop_size)]
    if seed_genome is not None:                 # warm start (memetic)
        pop[0] = _repair(list(seed_genome), C, rng)
    fit = np.array([evaluate(world, g, specs) for g in pop])
    history = [fit.max()]

    def tourney():
        cand = rng.integers(0, pop_size, tournament)
        return pop[cand[np.argmax(fit[cand])]]

    total = world.total_weight
    step = max(1, generations // 5)
    best_fit, stall, ran = fit.max(), 0, 0
    for gen in range(1, generations + 1):
        ran = gen
        order = np.argsort(fit)[::-1]
        new = [pop[i] for i in order[:elitism]]          # elitism
        while len(new) < pop_size:
            p1, p2 = tourney(), tourney()
            if rng.random() < crossover_rate:            # uniform crossover
                child = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(K)]
            else:
                child = list(p1)
            for i in range(K):                           # mutation
                if rng.random() < mutation_rate:
                    if rng.random() < 0.5:
                        child[i] = (int(rng.integers(C)), child[i][1])   # move
                    else:
                        th = (child[i][1] + rng.normal(0, 0.6)) % TWO_PI  # rotate
                        child[i] = (child[i][0], th)
            new.append(_repair(child, C, rng))
        pop = new
        fit = np.array([evaluate(world, g, specs) for g in pop])
        history.append(fit.max())
        if verbose and (gen % step == 0 or gen == generations):
            print(f"        [GA] gen {gen:>3}/{generations}  "
                  f"best {100*fit.max()/total:5.1f}%")
        if fit.max() > best_fit + min_delta:            # early stopping
            best_fit, stall = fit.max(), 0
        else:
            stall += 1
            if stall >= patience:
                if verbose:
                    print(f"        [GA] early stop at gen {gen} "
                          f"(no gain for {patience})")
                break

    best = pop[int(np.argmax(fit))]
    return _result("ga", world, best, specs, t0, history, iters=ran)


# 3. Particle Swarm Optimization (continuous x,y,theta; snap to candidate)
def pso(world, specs, swarm=30, iterations=200, w=0.7, c1=1.5, c2=1.5,
        seed=42, seed_genome=None, patience=20, min_delta=1e-6,
        verbose=True, **_):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    K = len(specs)
    xmin, xmax, ymin, ymax = world.bbox
    lo = np.tile([xmin, ymin, 0.0], K)
    hi = np.tile([xmax, ymax, TWO_PI], K)
    span = hi - lo

    def decode(pos):
        genome = []
        for i in range(K):
            x, y, th = pos[3 * i:3 * i + 3]
            genome.append((world.snap(x, y), th % TWO_PI))
        return genome

    def score(pos):
        return evaluate(world, decode(pos), specs)

    X = lo + rng.random((swarm, 3 * K)) * span
    if seed_genome is not None:                 # warm start one particle
        X[0] = np.concatenate([[world.cand_xy[c][0], world.cand_xy[c][1], th]
                               for c, th in seed_genome])
    V = (rng.random((swarm, 3 * K)) - 0.5) * span * 0.1
    pbest = X.copy()
    pbest_f = np.array([score(x) for x in X])
    g = int(np.argmax(pbest_f))
    gbest = pbest[g].copy(); gbest_f = pbest_f[g]
    history = [gbest_f]

    total = world.total_weight
    step = max(1, iterations // 5)
    stall, ran = 0, 0
    for it in range(1, iterations + 1):
        ran = it
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lo, hi)
        f = np.array([score(x) for x in X])
        better = f > pbest_f
        pbest[better] = X[better]; pbest_f[better] = f[better]
        g = int(np.argmax(pbest_f))
        improved = pbest_f[g] > gbest_f + min_delta
        if pbest_f[g] > gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        history.append(gbest_f)
        if verbose and (it % step == 0 or it == iterations):
            print(f"        [PSO] iter {it:>3}/{iterations}  "
                  f"best {100*gbest_f/total:5.1f}%")
        stall = 0 if improved else stall + 1            # early stopping
        if stall >= patience:
            if verbose:
                print(f"        [PSO] early stop at iter {it} "
                      f"(no gain for {patience})")
            break

    return _result("pso", world, decode(gbest), specs, t0, history, iters=ran)


# ---------------------------------------------------------------------------
def _result(name, world, genome, specs, t0, history=None, iters=None):
    cams = _cameras(world, genome, specs)
    return {
        "solver": name,
        "genome": genome,
        "cameras": cams,
        "coverage_weight": world.coverage_weight(cams),
        "coverage_pct": world.coverage_pct(cams),
        "overlap_pct": world.overlap_pct(cams),
        "objective": world.objective(cams),
        "runtime": time.perf_counter() - t0,
        "history": history,
        "iters": iters,
    }



# 4. Ant Colony Optimization  (subset selection, discretized angles)


_ACO_ANGLES = np.deg2rad(np.arange(0, 360, 45))          # 8 directions


def _node_heuristic(world, specs):
    """eta[c] = best single-camera coverage of candidate c (static heuristic)."""
    R = max(s[0] for s in specs)
    hf = max(s[1] for s in specs)
    eta = np.zeros(len(world.cand_ids))
    for c in range(len(world.cand_ids)):
        best = 0.0
        for th in _ACO_ANGLES:
            idx = world.camera_cover_idx(c, th, R, hf)
            best = max(best, world.dw[idx].sum())
        eta[c] = best
    return eta / (eta.max() + 1e-9)


def aco(world, specs, ants=20, iterations=200, alpha=1.0, beta=2.0, rho=0.1,
        q=1.0, seed=42, seed_genome=None, patience=20, min_delta=1e-6,
        verbose=True, **_):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    K, C = len(specs), len(world.cand_ids)
    A = len(_ACO_ANGLES)
    total = world.total_weight

    eta = _node_heuristic(world, specs)          # node desirability heuristic
    tau_sel = np.ones(C)                         # pheromone: pick this node
    tau_ang = np.ones((C, A))                    # pheromone: node's facing angle
    if seed_genome is not None:                  # warm-start pheromone
        for c, th in seed_genome:
            tau_sel[c] += 1.0
            tau_ang[c, int(round((th % (2*np.pi)) / (np.pi/4))) % A] += 1.0

    def build_ant():
        chosen = np.zeros(C, bool)
        genome = []
        base = (tau_sel ** alpha) * (eta ** beta)
        for i in range(K):
            p = base.copy()
            p[chosen] = 0
            s = p.sum()
            c = int(rng.integers(C)) if s <= 0 else int(rng.choice(C, p=p / s))
            chosen[c] = True
            pa = tau_ang[c] ** alpha
            a = int(rng.choice(A, p=pa / pa.sum()))
            genome.append((c, float(_ACO_ANGLES[a])))
        return genome

    # elitist warm-start: best-so-far begins at the greedy seed, so ACO's
    # returned solution can never be worse than greedy, and its pheromone is
    # reinforced toward that strong region from iteration 1.
    if seed_genome is not None:
        best_g = list(seed_genome)
        best_f = evaluate(world, best_g, specs)
    else:
        best_g, best_f = None, -1.0
    stall, ran = 0, 0
    history = []
    for it in range(1, iterations + 1):
        ran = it
        iter_best_g, iter_best_f = None, -1.0
        for _ in range(ants):
            g = build_ant()
            fval = evaluate(world, g, specs)
            if fval > iter_best_f:
                iter_best_f, iter_best_g = fval, g
        tau_sel *= (1 - rho); tau_ang *= (1 - rho)       # evaporation
        improved = iter_best_f > best_f + min_delta
        if iter_best_f > best_f:
            best_f, best_g = iter_best_f, iter_best_g
        for c, th in best_g:                              # deposit on best-so-far
            a = int(round((th % (2*np.pi)) / (np.pi/4))) % A
            tau_sel[c] += q * best_f / total
            tau_ang[c, a] += q * best_f / total
        history.append(best_f)
        if verbose and (it % max(1, iterations // 5) == 0 or it == iterations):
            print(f"        [ACO] iter {it:>3}/{iterations}  "
                  f"best {100*best_f/total:5.1f}%")
        stall = 0 if improved else stall + 1             # early stopping
        if stall >= patience:
            if verbose:
                print(f"        [ACO] early stop at iter {it}")
            break

    return _result("aco", world, best_g, specs, t0, history, iters=ran)


SOLVERS = {"greedy": greedy, "ga": ga, "pso": pso, "aco": aco}
