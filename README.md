# Dhaka Smart City Logistics & Planning: A Multi-Paradigm AI Benchmarking Suite

[![AI Paradigms](https://img.shields.io/badge/AI_Paradigms-Search_|_CSP_|_Metaheuristics_|_RL-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](#)
[![OSMnx](https://img.shields.io/badge/Map_Data-OpenStreetMap-orange.svg)](#)

This repository contains the complete codebase and academic report for a multi-paradigm study on urban planning and logistics optimization in a South-Asian megacity (Dhaka, Bangladesh). The project benchmarks four distinct Artificial Intelligence paradigms on a shared, data-grounded substrate using real OpenStreetMap road networks under realistic physical and policy constraints (dynamic traffic congestion, monsoon flooding, and fuel rationing).

---

## 🗺️ Shared Substrate: Dhaka Road Network

All tasks are evaluated on a real-world directed graph of Dhaka (Central Dhanmondi & surrounding arterials) extracted via `OSMnx`. The edge weights are dynamically computed using a **Context-Aware Cost Engine** that models:
*   **Monsoon Waterlogging:** Dynamic flood factors multiplying traversal costs on low-elevation roads.
*   **Stochastic Traffic Congestion:** Rush-hour traffic profiles varying by time-of-day.
*   **Multi-Criteria Preferences:** Personalized scaling for travel mode (walk, car, bus), security/safety sensitivity, and time efficiency.

---

## 📁 Repository Structure

```directory
├── search_algorithms.py     # Lab 1: BFS, DFS, UCS, A*, Weighted A*
├── cost_engine.py           # Lab 1: Shared Context-Aware Cost Model
├── scenarios.py             # Lab 1: Passenger Profiles (e.g., Safe, Fast)
├── main.py                  # Lab 1: Pathfinding Entrypoint
├── analysis.py              # Lab 1: Node Expansion Analytics
│
├── Lab2/
│   ├── csp_model.py         # Lab 2: Fuel Rationing Variables & Constraints
│   ├── backtracking.py      # Lab 2: Systematic Backtracking (MRV, LCV, FC)
│   ├── min_conflicts.py     # Lab 2: Min-Conflicts Local Search Repair
│   ├── gas_world.py         # Lab 2: Station Queue & Priority Modeling
│   ├── app.py               # Lab 2: Gradio UI for Interactive Allocations
│   └── csp_main.py          # Lab 2: CSP Solver Entrypoint
│
├── Lab3/
│   └── cctv/
│       ├── cctv_world.py    # Lab 3: 2D FOV Cones & Overlap Modeling
│       ├── solvers.py       # Lab 3: GA, PSO, ACO, and Greedy Placements
│       └── main.py          # Lab 3: CCTV Coverage Optimizer Entrypoint
│
├── Lab 4/
│   ├── delivery_env.py      # Lab 4: MDP Simulator (States, Stochastic Transitions)
│   ├── vi_agent.py          # Lab 4: Model-Based Value Iteration
│   ├── ql_agent.py          # Lab 4: Model-Free Q-Learning Agent
│   └── main.py              # Lab 4: Reinforcement Learning Entrypoint
│
├── Lab Report/
│   ├── main.tex             # LaTeX Source File
│   ├── references.bib       # Bibliography Database
│   └── main.pdf             # Compiled Academic Lab Report
│
├── requirements.txt         # Project Dependencies
└── README.md                # Repository Documentation
```

---

## 🚀 Lab Overview & Core Findings

### 🏎️ Lab 1: Context-Aware Pathfinding (Search)
*   **Objective:** Find optimal routing for distinct passenger profiles (e.g., solo female traveler at night avoiding dark alleys, or a commuter avoiding flooded roads).
*   **Algorithms:** Uniform Cost Search (UCS), A*, and Weighted A*.
*   **Key Insight:** Isolates how heuristic informativeness impacts search efficiency. When a heuristic aligns poorly with the dynamic multi-criteria cost surface (e.g., using straight-line Euclidean distance on a highly congested network), A*'s performance degrades toward UCS.

### ⛽ Lab 2: Policy-Regulated Fuel Rationing (CSP)
*   **Objective:** Allocate limited fuel to emergency, public, and private vehicles across refueling stations under capacity, priority, and topological routing constraints.
*   **Algorithms:** AC-3 (Arc Consistency), Backtracking with MRV/LCV/Forward Checking, and Min-Conflicts local search.
*   **Key Insight:** Demonstrates the scalability crossover point. Systematic backtracking performs optimally under loose-to-moderate constraint tightness but degrades exponentially near the phase transition, where Min-Conflicts local repair provides orders-of-magnitude faster solution times.

### 📸 Lab 3: CCTV Coverage Optimization (Metaheuristics)
*   **Objective:** Position a limited budget of security cameras at nodes and tune their continuous facing angles to maximize visible crime-prone areas while minimizing view overlap.
*   **Algorithms:** Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and Greedy.
*   **Key Insight:** Evaluates submodularity guarantees. Under submodular coverage (no overlap penalties), the Greedy heuristic easily meets the theoretical $(1-1/e)$ bound. When continuous angle variables and severe overlap penalties break submodularity, population-based metaheuristics (GA, PSO) significantly outperform Greedy by escaping local minima.

### 🚚 Lab 4: Multi-Stop Delivery Routing (Reinforcement Learning)
*   **Objective:** Teach a delivery agent to route through dynamic hazards (monsoon flooding and traffic spikes) to deliver high-priority parcels.
*   **Algorithms:** Model-Based Value Iteration (MDP) and Model-Free Q-Learning.
*   **Key Insight:** Quantifies the cost of model-free exploration. While the Q-learning agent successfully converges to the optimal policy learned by Value Iteration, it requires a $20\times$ premium in experience (thousands of episodes vs. dozens of value sweeps).

---

## 🛠️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/dhaka-smart-city-ai.git
    cd dhaka-smart-city-ai
    ```

2.  **Set Up a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Running the Code

### Task 1: Route Planning
```bash
python main.py
```
*(Runs A* and UCS on the Dhaka map, saving path visualizations to the root folder.)*

### Task 2: Fuel Allocation CSP
```bash
cd Lab2
python csp_main.py
```
*(Runs backtracking and local-search comparisons.)* Or run the interactive GUI:
```bash
python app.py
```

### Task 3: CCTV Placement Metaheuristics
```bash
cd Lab3/cctv
python main.py
```
*(Runs Greedy, GA, PSO, and ACO solvers and saves placement plots under `results/`.)*

### Task 4: Stochastic Delivery RL
```bash
cd "Lab 4"
python main.py
```
*(Trains Q-learning and Value Iteration agents, producing convergence and routing comparison plots.)*

---

## 📄 Academic Paper
The full academic analysis, mathematical formulations of the cost engines/MDP transitions, and detailed results are documented in:
👉 **[Lab Report/main.pdf](Lab%20Report/main.pdf)**
