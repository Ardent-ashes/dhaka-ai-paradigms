# Gas Allocation in Dhaka — A Constraint Satisfaction Problem

A CSP-based simulation that decides which user goes to which gas station during a fuel crisis in Dhaka. Built on real OpenStreetMap road data, with realistic constraints (capacity limits, plate-parity rationing, household pairing, peak-hour CNG closure, etc.) and two solver methods: **Backtracking with heuristics** and **Min-Conflicts local search**.

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [CSP Formulation](#csp-formulation)
3. [Project Structure](#project-structure)
4. [Constraints](#constraints)
5. [Solver Methods](#solver-methods)
6. [How to Run](#how-to-run)
7. [Output and Visualization](#output-and-visualization)

---

## Problem Overview

Imagine a fuel crisis in Dhaka. There are limited gas stations, each with finite capacity. There are many users (private cars, public buses, emergency vehicles) who all need fuel. The government imposes rationing policies — odd/even plate days, peak-hour CNG restrictions, strategic reserves for emergencies, household pairing rules, and so on.

**The question**: who should go to which station such that every constraint is satisfied and the total cost (distance traveled + load imbalance) is minimized?

This is a classic **Constraint Satisfaction Problem (CSP)**, and the project demonstrates how to model and solve it using textbook AI techniques.

---

## CSP Formulation

A CSP needs three things:

| Component | In this project |
|---|---|
| **Variables** | One per user. Each user must be assigned exactly one station. |
| **Domains** | The set of stations each user *could* go to (filtered by reachability, vehicle class, rationing, etc.) |
| **Constraints** | Rules that must hold across the assignment (9 in total, see below). |

A solution is a complete assignment `{user_id -> station_id}` that satisfies every active constraint. There is also a soft objective — total travel distance plus load imbalance — that the solver tries to minimize.

---

## Project Structure

```
data_loader.py     →  Loads Dhaka's OSM road graph (cached locally)
       ↓
gas_world.py       →  Places stations + users on the graph,
                       computes per-vehicle-class distance tables
                       using reverse Dijkstra
       ↓
csp_model.py       →  Defines the CSP — domains, constraints, AC-3
       ↓
    ┌──┴──┐
backtracking.py   min_conflicts.py    ← two solvers
    └──┬──┘
       ↓
visualize.py       →  Plots: assignment maps, station loads,
                       convergence curves, comparison bars
       ↓
csp_main.py        →  Interactive CLI for manual experimentation
experiments.py     →  Automated batch experiments for the report
```

### File-by-file responsibilities

- **`data_loader.py`** — Downloads Dhaka's full road network from OpenStreetMap once and caches it as `cache/dhaka_all.graphml`. The graph has nodes (intersections) and edges (road segments) with length and road-type attributes.

- **`gas_world.py`** — The "world" the CSP operates on. It places real gas stations (from OSM `amenity=fuel`) onto the graph, generates synthetic users with vehicle class / plate parity / remaining fuel / fuel demand, and pre-computes distances using **reverse Dijkstra** (one run per station, per vehicle class — far cheaper than running forward Dijkstra per user).

- **`csp_model.py`** — The heart of the CSP. Defines initial domains via unary constraints, runs AC-3 for the binary household constraint, and provides consistency checks (partial during search, complete at the end). Also has a violation counter for Min-Conflicts.

- **`backtracking.py`** — Method 1: systematic recursive search. Uses three textbook heuristics (MRV, Degree, LCV) plus Forward Checking inference.

- **`min_conflicts.py`** — Method 2: local search heuristic. Starts from a random complete assignment and iteratively moves conflicted users to minimize-violation stations. Random restart on plateau.

- **`visualize.py`** — Renders the assignment on the Dhaka map (color-coded by station), bar charts of station loads vs. capacity, convergence curves for Min-Conflicts, and side-by-side comparison plots.

- **`csp_main.py`** — Interactive CLI where the user picks the scenario (bbox, number of users, time slot, seed), toggles constraints on/off, tunes policy knobs, and selects which solver(s) to run.

- **`experiments.py`** — Automated runs A/B/C/demo that produce all the figures and CSVs needed for the report (head-to-head on small instance, scaling sweep, full-Dhaka realistic case, slide-aligned demo).

---

## Constraints

The project implements **9 constraints**, grouped by type. Numbering follows the course's slide deck (some numbers are skipped intentionally — those weren't part of this implementation).

### Unary constraints (single user, applied during domain filtering)

#### U1 — Reachability
A user can only be assigned to a station they can actually reach with their remaining fuel.

```
distance(user, station) ≤ user.fuel_left_km × 1000
```

The distance is jam-aware and road-size-aware (computed in `gas_world.py`).

#### U2 — Emergency-Only Stations
Some stations (≈ 8%) are reserved exclusively for emergency vehicles (ambulance, fire). Private and public vehicles cannot be assigned to them.

#### G1 — Plate-Parity Rationing
If today's policy is "odd-only", civilians with even plates get an **empty domain** (they cannot fuel today). Emergency vehicles are exempt. `today_parity` can be `"odd"`, `"even"`, or `"any"`.

#### G6 — Peak-Hour CNG Closure
During peak traffic slots (7–10 AM, 5–9 PM), CNG stations are closed for non-emergency vehicles. Mirrors a real Dhaka rule meant to keep public transport on the road.

### Binary constraint (between two users — AC-3 runs here)

#### B1 — Household Constraint
Users in the same household must be assigned to **different** stations. This prevents one family from gaming the rationing system. Households are randomly generated using `household_fraction` (default 20% of users grouped into pairs).

This is the only constraint that benefits from **arc consistency (AC-3)** — it prunes domains before search begins.

### Global / n-ary constraints (station-wide aggregates)

#### G3 — Public Transport Reserve
Each station reserves a fraction (default 60%) of its civilian capacity for public transport. Private vehicles can use at most the remaining 40%. This keeps buses and CNGs running so the city doesn't grind to a halt.

#### G4 — Strategic Reserve
A fraction (default 10%) of every station's total capacity is locked for emergency vehicles only. Civilians (private + public) can only use up to `cap × (1 - 0.10) = 90%` of the station.

#### G8 — Queue Cap
No more than K users (default 8) can be queued at the same station. Larger queues spill onto the road and create traffic chaos.

#### G9 — Minimum Service Quota
Every station must serve at least M users (default 1). Prevents the solver from cramming everyone into one or two convenient stations and starving the rest. **Only checked at the end** of a complete assignment, since partial assignments will naturally fail it.

### Summary table

| ID | Type | Policy knob | When applied |
|---|---|---|---|
| **U1** Reachability | Unary | — | Domain construction |
| **U2** Emergency-only | Unary | — | Domain construction |
| **G1** Plate parity | Unary | `today_parity` | Domain construction |
| **G3** Public reserve | Global | `public_reserve_fraction` | Partial + complete check |
| **G4** Strategic reserve | Global | `strategic_reserve_fraction` | Partial + complete check |
| **G6** Peak CNG closure | Unary | — | Domain construction |
| **G8** Queue cap | Global | `queue_cap` | Partial + complete check |
| **G9** Min service quota | Global | `min_service_quota` | Complete check only |
| **B1** Household | Binary | `household_fraction` | AC-3 + partial check |

### Soft objective (what gets minimized among valid solutions)

```
total_cost = w_distance × Σ distance(user, assigned_station)
           + w_load     × variance(station_loads)
```

The solver may find many valid assignments; the cost function picks the best.

---

## Solver Methods

### Method 1: Backtracking with Heuristics

A classic systematic search: try assigning user 1 to its first option, then user 2, and so on. If at any point a constraint is violated, undo the most recent decision and try a different value — i.e., **backtrack**.

Pure backtracking is exponential. To make it tractable, three textbook heuristics are layered on top:

- **MRV (Minimum Remaining Values)** — pick the variable whose domain is smallest. The intuition is that constrained variables fail fastest, so fail early and prune the search tree.
- **Degree heuristic** — tiebreaker for MRV. Among variables with equal domain size, pick the one connected to the most unassigned neighbors (it has the highest "impact").
- **LCV (Least Constraining Value)** — given the chosen variable, try its values in the order that constrains other variables the least, maximizing the chance of success.

Plus **Forward Checking (FC)**: after every assignment, prune the chosen value from the domains of household neighbors. If any neighbor's domain becomes empty, fail immediately — don't go deeper.

Backtracking is **complete** — if a solution exists, it will find it (given enough time). But on large instances, "enough time" can be infeasible.

### Method 2: Min-Conflicts Local Search

A radically different philosophy: start from a **random complete assignment** (every feasible user gets some station from their domain), then iteratively improve.

The algorithm:

1. Pick a conflicted user at random.
2. Move them to the station that minimizes the total violation count (ties broken randomly).
3. Repeat until violations = 0, or until stuck on a plateau, then random restart.

Min-Conflicts is **incomplete** — it has no guarantee of finding a solution, and can get stuck in local minima. But it scales beautifully: it routinely solves problems too large for backtracking (the classic example: 1 million queens in a few seconds).

It also has a useful property: even if it fails, it returns the **best-so-far** assignment, which is often a useful approximate solution.

### When to use which

| Scenario | Better solver |
|---|---|
| Small instance (<30 users), need optimality | Backtracking |
| Large instance (100+ users), need a fast good-enough answer | Min-Conflicts |
| Want to prove correctness | Backtracking |
| Want to demonstrate scaling | Min-Conflicts |

The project's "demo" run in `experiments.py` runs Backtracking on a small Dhanmondi scenario and Min-Conflicts on full Dhaka — exactly to showcase this trade-off.

---

## How to Run

### Interactive CLI

```bash
python csp_main.py
```

This prompts the user for:
- Number of users, time slot, random seed
- Area preset (Dhanmondi / Mohakhali / Mirpur / Old Dhaka / Full Dhaka)
- Maximum number of stations
- Which constraints to toggle on/off
- Policy parameters (parity, queue cap, reserve fractions, household fraction)
- Which solver(s) to run

A recommended starting choice is the **three-way comparison**: Naive BT (no heuristics) vs. Heuristic BT vs. Min-Conflicts on the same scenario. It shows directly how much the heuristics help and where Min-Conflicts dominates.

### Standalone batch experiments

```bash
python experiments.py --run all
```

Runs four predefined experiments:
- **Run A** — Head-to-head BT vs. Min-Conflicts on a small Dhanmondi scenario.
- **Run B** — Scaling sweep: N = 10, 20, 40, 80, 160 users. Shows where BT times out and MC keeps going.
- **Run C** — Realistic full-Dhaka case (150 users, 30 stations), Min-Conflicts only.
- **Run demo** — Slide-aligned: BT on small portion (Dhanmondi, 12 users) vs. Min-Conflicts on whole Dhaka (150 users), with a bonus "BT attempts the large case and times out" demonstration.

Each module also has its own `if __name__ == "__main__":` smoke test for quick standalone testing during development.

### Caching note

The first run downloads the Dhaka OSM graph (~30 seconds) and queries for fuel stations. Both are cached to `cache/`. Subsequent runs are fast.

---

## Output and Visualization

Every run writes outputs to `cache/plots/<run_id>/`:

- **Assignment map** — Dhaka road graph with stations marked as colored squares, assigned users as colored dots (matching their station), and unassignable users as red X marks. Lines connect each user to their station.
- **Station loads** — Stacked bar chart showing private / public / emergency demand at each station vs. its total capacity, civilian capacity, and private cap.
- **Convergence curve** (Min-Conflicts only) — Violation count over iterations. Should drop steeply at first, then plateau toward zero.
- **Comparison bars** — Side-by-side cost / runtime / search-effort comparisons across solver variants.
- **Scaling plot** (Run B) — Runtime vs. N on a log scale, showing where each solver's curve takes off.

A CSV log (`cache/csp_history.csv`) records every run: scenario parameters, solver, found/cost/runtime, search effort. This is what feeds the report's tables.

---

## Key Concepts Demonstrated

This project hits all the classic CSP topics from the AI textbook:

- **Variables, domains, constraints** — the basic CSP formulation.
- **Unary / binary / global constraints** — and how they're handled differently.
- **Arc consistency (AC-3)** — preprocessing that prunes domains before search.
- **Backtracking search** — depth-first systematic search with backtrack on failure.
- **MRV + Degree heuristic** — smart variable ordering.
- **LCV** — smart value ordering.
- **Forward Checking** — lightweight inference during search.
- **Local search (Min-Conflicts)** — a complement to systematic search for large problems.
- **Soft constraints / objective functions** — going beyond satisfaction to optimization.
- **Heuristic ablation** — measuring what each heuristic actually contributes.
- **Scaling behavior** — empirical evidence for when each method works.

All wrapped around a concrete, geographically grounded scenario (real Dhaka roads, real OSM stations), which makes the trade-offs tangible.
