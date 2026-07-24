import itertools
import numpy as np
import networkx as nx

from data_loader import load_dhaka_graph

# preset study areas (centre lat, lon)
AREAS = {
    "dhanmondi": (23.7461, 90.3742),
    "tsc":       (23.7305, 90.3918),   # DU / TSC
    "gulshan":   (23.7925, 90.4078),
    "mirpur":    (23.8041, 90.3667),
    "old_dhaka": (23.7104, 90.4074),
}

# assumed speed (km/h) by OSM highway type -> travel time
SPEED_KMH = {
    "motorway": 45, "trunk": 45, "primary": 40, "secondary": 32,
    "tertiary": 25, "unclassified": 22, "residential": 18,
    "living_street": 12, "service": 12,
}
DEFAULT_SPEED = 20


JAM_TYPES   = {"motorway", "trunk", "primary", "secondary", "tertiary",
               "unclassified",
               "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
FLOOD_TYPES = {"residential", "living_street", "service"}


def _htype(hw):
    return hw[0] if isinstance(hw, list) else hw


class DeliveryEnv:
    def __init__(self, area="dhanmondi", center=None, n_nodes=150, n_targets=3,
                 n_urgent=1, jam_frac=0.18, flood_frac=0.12, gamma=0.95,
                 seed=42, G=None, verbose=True,
                 node_depot=None, node_targets=None):
        """
        node_depot   : int index into sorted node list (from --show-map). If None → random.
        node_targets : list of int indices. If None → random.
        """
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        # reward parameters
        self.C_hazard = 50.0      # penalty for a stall or a traffic divert
        self.R_target = 200.0     # reaching a normal delivery stop
        self.R_urgent = 400.0     # reaching an URGENT delivery stop
        self.R_done = 1000.0      # all deliveries complete (terminal)

        lat, lon = center if center else AREAS[area]
        full = G if G is not None else load_dhaka_graph()
        self.graph = self._build_subgraph(full, lat, lon, n_nodes)
        self.nodes = sorted(list(self.graph.nodes))   # sorted → consistent index ↔ node
        if verbose:
            print(f"[env] {area}: sub-graph {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")

        self._index_edges()
        self._inject_hazards(jam_frac, flood_frac, verbose)

        # pick depot + delivery targets (user-specified OR random)
        if node_depot is not None and node_targets is not None:
            self.start   = self.nodes[node_depot]
            self.targets = tuple(self.nodes[i] for i in node_targets)
            n_targets    = len(self.targets)          # override so states() is correct
        else:
            choice       = list(self.rng.permutation(self.nodes))
            self.start   = choice[0]
            self.targets = tuple(choice[1:1 + n_targets])

        self.urgent = set(self.targets[:n_urgent])
        if verbose:
            depot_idx   = self.nodes.index(self.start)
            target_idxs = [self.nodes.index(t) for t in self.targets]
            print(f"[env] depot=#{depot_idx}({self.start}), "
                  f"targets=#{target_idxs}({list(self.targets)}), "
                  f"urgent=#{[self.nodes.index(u) for u in self.urgent]}")
        self._n_targets = n_targets   # store for states()

    # ------------------------------------------------------------------ setup
    def _build_subgraph(self, G, lat, lon, n_nodes):
        ids = np.array(list(G.nodes))
        xs = np.array([G.nodes[n]["x"] for n in ids])
        ys = np.array([G.nodes[n]["y"] for n in ids])
        d = (xs - lon) ** 2 + (ys - lat) ** 2
        near = ids[np.argsort(d)[:n_nodes * 3]]
        H = G.subgraph(near).copy()
        # keep the largest strongly-connected core so every stop is reachable
        core = max(nx.strongly_connected_components(H), key=len)
        H = H.subgraph(core).copy()
        if H.number_of_nodes() > n_nodes:                 # trim, keep connected
            ids2 = np.array(list(H.nodes))
            xs2 = np.array([H.nodes[n]["x"] for n in ids2])
            ys2 = np.array([H.nodes[n]["y"] for n in ids2])
            d2 = (xs2 - lon) ** 2 + (ys2 - lat) ** 2
            keep = ids2[np.argsort(d2)[:n_nodes]]
            H = H.subgraph(keep).copy()
            H = H.subgraph(max(nx.strongly_connected_components(H), key=len)).copy()
        return H

    def _index_edges(self):
        """Per directed (u,v): length (m), road type, travel time (min)."""
        self.neighbors = {}
        self.length = {}
        self.rtype = {}
        self.tt = {}
        for u in self.graph.nodes:
            succ = list(dict.fromkeys(self.graph.successors(u)))
            self.neighbors[u] = succ
            for v in succ:
                data = min(self.graph[u][v].values(), key=lambda d: d.get("length", 1))
                length = float(data.get("length", 1.0))
                ht = _htype(data.get("highway"))
                spd = SPEED_KMH.get(ht, DEFAULT_SPEED)
                self.length[(u, v)] = length
                self.rtype[(u, v)] = ht
                self.tt[(u, v)] = (length / 1000.0) / spd * 60.0   # minutes

    def _inject_hazards(self, jam_frac, flood_frac, verbose):
        self.hazard = {}
        jam_pool, flood_pool = [], []
        for (u, v), ht in self.rtype.items():
            self.hazard[(u, v)] = "normal"
            if ht in JAM_TYPES:
                jam_pool.append((u, v))
            elif ht in FLOOD_TYPES:
                flood_pool.append((u, v))
        for pool, frac, tag in [(jam_pool, jam_frac, "jam"),
                                (flood_pool, flood_frac, "flood")]:
            if pool:
                k = max(1, int(len(pool) * frac))
                for i in self.rng.choice(len(pool), k, replace=False):
                    self.hazard[pool[i]] = tag
        if verbose:
            n_jam = sum(v == "jam" for v in self.hazard.values())
            n_fl = sum(v == "flood" for v in self.hazard.values())
            print(f"[env] hazards: {n_jam} jam edges, {n_fl} flooded edges")

    # ------------------------------------------------------------------- MDP
    def reset(self):
        return (self.start, frozenset(self.targets))

    def is_terminal(self, s):
        return len(s[1]) == 0

    def actions(self, s):
        return self.neighbors[s[0]]

    def states(self):
        """All valid states: current node + any subset of targets (node not
        itself an undelivered target)."""
        subs = []
        for r in range(len(self.targets) + 1):
            for c in itertools.combinations(self.targets, r):
                subs.append(frozenset(c))
        for node in self.nodes:
            for U in subs:
                if node not in U:
                    yield (node, U)

    def transitions(self, s, a):
        """Return list of (prob, next_state, reward)."""
        c, U = s
        nbrs = self.neighbors[c]
        haz = self.hazard[(c, a)]
        others = [n for n in nbrs if n != a]

        if haz == "flood":                       # 50% cross, 50% engine stall
            outcomes = [(0.5, a)] + ([(0.5, c)] if True else [])
        elif haz == "jam":                       # 70% ok, 30% diverted
            outcomes = ([(0.7, a)] +
                        [(0.3 / len(others), o) for o in others]) if others \
                        else [(1.0, a)]
        else:                                    # normal: 95% ok, 5% slip
            outcomes = ([(0.95, a)] +
                        [(0.05 / len(others), o) for o in others]) if others \
                        else [(1.0, a)]

        result = []
        for p, cp in outcomes:
            stalled = (cp == c)
            diverted = (cp != a) and not stalled
            if stalled:
                r = -self.C_hazard                       # stuck, no progress
            else:
                r = -self.tt[(c, cp)]                     # travel time cost
                if diverted:
                    r -= self.C_hazard                    # penalised divert
            Up = U
            if cp in U:                                   # delivered a stop!
                Up = U - {cp}
                r += self.R_urgent if cp in self.urgent else self.R_target
                if len(Up) == 0:
                    r += self.R_done                      # all done (terminal)
            result.append((p, (cp, Up), r))
        return result

    def step(self, s, a):
        """Sample one stochastic outcome (for Q-learning)."""
        outs = self.transitions(s, a)
        i = self.rng.choice(len(outs), p=[p for p, _, _ in outs])
        _, sp, r = outs[i]
        return sp, r, self.is_terminal(sp)

    # ------------------------------------------------------ rollout for plots
    def evaluate_policy(self, policy, max_steps=300):
        """Deterministic route reward: follow policy assuming intended moves."""
        s = self.reset(); total = 0.0; seen = set()
        for _ in range(max_steps):
            if self.is_terminal(s) or s in seen:
                break
            seen.add(s)
            a = policy.get(s)
            if a is None:
                break
            succ = [(p, sp, r) for p, sp, r in self.transitions(s, a)
                    if sp[0] == a]
            if not succ:
                break
            _, sp, r = succ[0]
            total += r; s = sp
        return total

    def greedy_route(self, policy, max_steps=300):
        """Follow a policy assuming intended moves succeed -> node sequence."""
        s = self.reset()
        route = [s[0]]
        seen = set()
        for _ in range(max_steps):
            if self.is_terminal(s) or s in seen:
                break
            seen.add(s)
            a = policy.get(s)
            if a is None:
                break
            c, U = s
            Up = U - {a} if a in U else U
            s = (a, Up)
            route.append(a)
        return route
