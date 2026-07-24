from collections import deque
from dataclasses import dataclass
import heapq
import math
import time

from cost_engine import edge_cost, heuristic_distance, heuristic_edge_cost


@dataclass
class SearchResult:
    algorithm: str
    path: list
    found: bool
    total_cost: float
    distance_m: float
    nodes_expanded: int
    nodes_popped: int
    revisit_count: int
    runtime_ms: float


def _min_edge_data(graph, u, v):
    data = graph.get_edge_data(u, v)
    if not data:
        return None
    if "length" in data:
        return data
    best_key = min(data.keys(), key=lambda k: float(data[k].get("length", 1.0)))
    return data[best_key]


def _neighbors(graph, node):
    return list(graph.successors(node)) if graph.is_directed() else list(graph.neighbors(node))


def _path_distance(graph, path):
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        data = _min_edge_data(graph, path[i], path[i + 1])
        if data:
            total += float(data.get("length", 1.0))
    return total


def _path_scenario_cost(graph, path, scenario):
    """Sum of edge_cost along path — same unit as UCS/A* total_cost."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        data = _min_edge_data(graph, path[i], path[i + 1])
        if data:
            total += edge_cost(data, scenario)
    return total


def _reconstruct(parent, goal):
    if goal not in parent:
        return []
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def build_admissible_heuristic_cache(graph, goal, scenario):
    dist = {goal: 0.0}
    heap = [(0.0, goal)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if graph.is_directed():
            sources = graph.predecessors(u)
        else:
            sources = graph.neighbors(u)
        for v in sources:
            data = _min_edge_data(graph, v, u)
            if not data:
                continue
            w = heuristic_edge_cost(data, scenario)
            if not math.isfinite(w):
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _h_value(graph, node, goal, h_cache):
    if h_cache is not None:
        return h_cache.get(node, math.inf)
    return heuristic_distance(graph, node, goal)


def _make_result(name, graph, path, total_cost, expanded, popped, revisits, start_time):
    return SearchResult(
        algorithm=name,
        path=path,
        found=bool(path),
        total_cost=total_cost if path else math.inf,
        distance_m=_path_distance(graph, path),
        nodes_expanded=expanded,
        nodes_popped=popped,
        revisit_count=revisits,
        runtime_ms=(time.perf_counter() - start_time) * 1000.0,
    )


def bfs(graph, start, goal, scenario):
    t0 = time.perf_counter()
    q = deque([start])
    parent = {start: None}
    visited = set()
    expanded = popped = revisits = 0

    while q:
        node = q.popleft()
        popped += 1
        if node in visited:
            revisits += 1
            continue
        visited.add(node)
        expanded += 1
        if node == goal:
            path = _reconstruct(parent, goal)
            cost = _path_scenario_cost(graph, path, scenario)
            return _make_result("BFS", graph, path, cost, expanded, popped, revisits, t0)

        for nbr in _neighbors(graph, node):
            if nbr not in parent:
                data = _min_edge_data(graph, node, nbr)
                if data and math.isfinite(edge_cost(data, scenario)):
                    parent[nbr] = node
                    q.append(nbr)

    return _make_result("BFS", graph, [], math.inf, expanded, popped, revisits, t0)


def dfs(graph, start, goal, scenario):
    t0 = time.perf_counter()
    stack = [start]
    parent = {start: None}
    visited = set()
    expanded = popped = revisits = 0

    while stack:
        node = stack.pop()
        popped += 1
        if node in visited:
            revisits += 1
            continue
        visited.add(node)
        expanded += 1
        if node == goal:
            path = _reconstruct(parent, goal)
            cost = _path_scenario_cost(graph, path, scenario)
            return _make_result("DFS", graph, path, cost, expanded, popped, revisits, t0)

        for nbr in reversed(_neighbors(graph, node)):
            if nbr not in parent:
                data = _min_edge_data(graph, node, nbr)
                if data and math.isfinite(edge_cost(data, scenario)):
                    parent[nbr] = node
                    stack.append(nbr)

    return _make_result("DFS", graph, [], math.inf, expanded, popped, revisits, t0)


def iddfs(graph, start, goal, scenario, max_depth=40):
    t0 = time.perf_counter()
    total_expanded = total_popped = total_revisits = 0

    for depth_limit in range(max_depth + 1):
        stack = [(start, 0, [start])]
        seen_at_depth = set()

        while stack:
            node, depth, path = stack.pop()
            total_popped += 1

            key = (node, depth)
            if key in seen_at_depth:
                total_revisits += 1
                continue
            seen_at_depth.add(key)
            total_expanded += 1

            if node == goal:
                cost = _path_scenario_cost(graph, path, scenario)
                return _make_result("IDDFS", graph, path, cost,
                                    total_expanded, total_popped, total_revisits, t0)

            if depth >= depth_limit:
                continue

            for nbr in reversed(_neighbors(graph, node)):
                if nbr not in path:  
                    data = _min_edge_data(graph, node, nbr)
                    if data and math.isfinite(edge_cost(data, scenario)):
                        stack.append((nbr, depth + 1, path + [nbr]))

    return _make_result("IDDFS", graph, [], math.inf,
                        total_expanded, total_popped, total_revisits, t0)


def ucs(graph, start, goal, scenario):
    return _weighted_search(graph, start, goal, scenario, "UCS")


def greedy_best_first(graph, start, goal, scenario, h_cache=None):
    t0 = time.perf_counter()
    heap = [(_h_value(graph, start, goal, h_cache), start)]
    parent = {start: None}
    visited = set()
    expanded = popped = revisits = 0
    cost_so_far = {start: 0.0}

    while heap:
        _, node = heapq.heappop(heap)
        popped += 1
        if node in visited:
            revisits += 1
            continue
        visited.add(node)
        expanded += 1

        if node == goal:
            path = _reconstruct(parent, goal)
            return _make_result(
                "Greedy Best-First",
                graph,
                path,
                cost_so_far.get(goal, math.inf),
                expanded,
                popped,
                revisits,
                t0,
            )

        for nbr in _neighbors(graph, node):
            data = _min_edge_data(graph, node, nbr)
            if not data:
                continue
            step = edge_cost(data, scenario)
            if not math.isfinite(step):
                continue
            if nbr not in parent:
                parent[nbr] = node
                cost_so_far[nbr] = cost_so_far[node] + step
                heapq.heappush(heap, (_h_value(graph, nbr, goal, h_cache), nbr))

    return _make_result(
        "Greedy Best-First", graph, [], math.inf, expanded, popped, revisits, t0
    )


def astar(graph, start, goal, scenario, h_cache=None):
    return _weighted_search(
        graph, start, goal, scenario, "A*", heuristic_weight=1.0, h_cache=h_cache
    )


def weighted_astar(graph, start, goal, scenario, weight=1.7, h_cache=None):
    return _weighted_search(
        graph,
        start,
        goal,
        scenario,
        "Weighted A*",
        heuristic_weight=weight,
        h_cache=h_cache,
    )


def _weighted_search(
    graph,
    start,
    goal,
    scenario,
    name,
    heuristic_weight=0.0,
    h_cache=None,
):
    t0 = time.perf_counter()
    heap = [(0.0, start)]
    parent = {start: None}
    g_cost = {start: 0.0}
    closed = set()
    expanded = popped = revisits = 0

    while heap:
        _, node = heapq.heappop(heap)
        popped += 1
        if node in closed:
            revisits += 1
            continue
        closed.add(node)
        expanded += 1

        if node == goal:
            path = _reconstruct(parent, goal)
            return _make_result(
                name,
                graph,
                path,
                g_cost.get(goal, math.inf),
                expanded,
                popped,
                revisits,
                t0,
            )

        for nbr in _neighbors(graph, node):
            data = _min_edge_data(graph, node, nbr)
            if not data:
                continue
            step = edge_cost(data, scenario)
            if not math.isfinite(step):
                continue

            new_cost = g_cost[node] + step
            if new_cost < g_cost.get(nbr, math.inf):
                g_cost[nbr] = new_cost
                parent[nbr] = node
                h = _h_value(graph, nbr, goal, h_cache)
                f = new_cost + (heuristic_weight * h)
                heapq.heappush(heap, (f, nbr))

    return _make_result(name, graph, [], math.inf, expanded, popped, revisits, t0)
