"""
solver.py — Core optimization algorithms for learning path construction.

Two algorithms are implemented:
  1. Greedy       — deterministic baseline, selects by utility/hour ratio
  2. Hill Climbing — local search metaheuristic with random restarts

Both return a Result object for side-by-side experimental comparison.
"""

import json
import random
from collections import deque
from dataclasses import dataclass


# Resources below this boosted utility threshold are not selected as targets.
# They may still appear as forced prerequisites of high-utility resources.
MIN_UTILITY = 4.0


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Resource:
    id: str
    name: str
    description: str
    duration_hours: int
    prerequisites: list
    utility: float = 0.0


@dataclass
class Result:
    ordered_path: list
    total_utility: float
    total_hours: int
    algorithm: str
    iterations: int = 0
    restarts: int = 0

    def summary(self) -> str:
        lines = [
            f"Algorithm  : {self.algorithm}",
            f"Total hours: {self.total_hours}h",
            f"Utility    : {self.total_utility:.2f}",
            f"Resources  : {len(self.ordered_path)}",
        ]
        if self.iterations:
            lines.append(f"Iterations : {self.iterations} ({self.restarts} restarts)")
        lines.append("Path:")
        for i, r in enumerate(self.ordered_path, 1):
            lines.append(f"  {i}. [{r.duration_hours}h | {r.utility:.1f}/10] {r.name}")
        return "\n".join(lines)


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_resources(dataset_path: str, scores: dict) -> list:
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Resource(
            id=item["id"],
            name=item["name"],
            description=item["description"],
            duration_hours=item["duration_hours"],
            prerequisites=item["prerequisites"],
            utility=scores.get(item["id"], 5.0),
        )
        for item in raw
    ]


def build_index(resources: list) -> dict:
    return {r.id: r for r in resources}

def apply_dependency_boost(resources: list, index: dict, alpha: float = 0.3) -> None:
    """
    Boost foundational resources based on their transitive descendants.

    The LLM undervalues foundational resources (e.g. Python básico) because it
    scores direct relevance, not structural importance. This corrects that:
    for each resource, find every resource that transitively depends on it,
    then boost by alpha * max_utility_among_those_descendants.

    Why max (not average): a resource is worth learning if it unlocks even ONE
    highly valuable advanced topic.

    Scores are capped at 10.0 and modified in place. Raw LLM scores are
    preserved in scores.json — the boost is applied at runtime only.

    Args:
        alpha: boost weight. Default 0.3 means a resource whose best descendant
               scores 9.0 gets a +2.7 boost before the cap.
    """
    for r in resources:
        descendants = [
            other for other in resources
            if r.id != other.id and r.id in compute_closure(other.id, index)
        ]
        if descendants:
            max_desc_utility = max(index[d.id].utility for d in descendants)
            r.utility = min(10.0, r.utility + alpha * max_desc_utility)



# ── Graph utilities ────────────────────────────────────────────────────────────

def compute_closure(resource_id: str, index: dict) -> set:
    """
    Return the full dependency closure: the resource itself plus every ancestor
    it transitively depends on. Uses iterative DFS to avoid recursion limits.
    """
    closure = set()
    stack = [resource_id]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        for prereq in index[current].prerequisites:
            if prereq not in closure:
                stack.append(prereq)
    return closure


def topological_sort(selection: set, index: dict) -> list:
    """
    Order a dependency-closed selection using Kahn's algorithm (BFS-based).
    Guarantees every prerequisite appears before the resources that need it.
    """
    in_degree = {rid: 0 for rid in selection}
    for rid in selection:
        for prereq in index[rid].prerequisites:
            if prereq in selection:
                in_degree[rid] += 1

    queue = deque(rid for rid in selection if in_degree[rid] == 0)
    ordered = []

    while queue:
        current = queue.popleft()
        ordered.append(index[current])
        for rid in selection:
            if current in index[rid].prerequisites:
                in_degree[rid] -= 1
                if in_degree[rid] == 0:
                    queue.append(rid)

    if len(ordered) != len(selection):
        raise ValueError("Cycle detected — dataset validation must have been skipped.")

    return ordered


def _hours(selection: set, index: dict) -> int:
    return sum(index[rid].duration_hours for rid in selection)


def _utility(selection: set, index: dict) -> float:
    return sum(index[rid].utility for rid in selection)


# ── Algorithm 1: Greedy ────────────────────────────────────────────────────────

def greedy_solver(resources: list, budget: int) -> Result:
    """
    Greedy algorithm: at each step, add the resource whose dependency closure
    gives the best utility-per-hour ratio, until nothing more fits.

    Key fix: we pre-filter candidates to those whose closure actually fits in
    the remaining budget before ranking. This prevents an infinite loop where
    the "best" candidate repeatedly fails the budget check.

    Connection to lectures:
        CSP    → hard constraints checked at every step
        Greedy → "problem-specific heuristic" from the metaheuristics taxonomy
    """
    index = build_index(resources)
    selection: set = set()

    while True:
        remaining = budget - _hours(selection, index)

        # Build list of feasible candidates: unselected, closure fits in budget
        feasible = []
        for r in resources:
            if r.id in selection:
                continue
            closure = compute_closure(r.id, index)
            new_ids = closure - selection
            new_hours = sum(index[rid].duration_hours for rid in new_ids)
            new_utility = sum(index[rid].utility for rid in new_ids)
            if new_hours > 0 and new_hours <= remaining and r.utility >= MIN_UTILITY:
                feasible.append((r, new_ids, new_hours, new_utility))

        if not feasible:
            break   # nothing fits anymore — we're done

        # Pick the candidate with the best utility/hour ratio for the NEW resources
        feasible.sort(key=lambda x: x[3] / x[2], reverse=True)
        best_r, best_new_ids, _, _ = feasible[0]

        closure = compute_closure(best_r.id, index)
        selection |= closure

    ordered = topological_sort(selection, index)
    return Result(
        ordered_path=ordered,
        total_utility=_utility(selection, index),
        total_hours=_hours(selection, index),
        algorithm="Greedy",
    )


# ── Algorithm 2: Hill Climbing with random restarts ───────────────────────────

def _random_valid_solution(resources: list, index: dict,
                           budget: int, rng: random.Random) -> set:
    """
    Generate a random valid starting solution by shuffling resources and
    adding each one (with its closure) while the budget allows.
    Called once per hill climbing restart to explore different search regions.
    """
    selection: set = set()
    shuffled = resources[:]
    rng.shuffle(shuffled)
    for r in shuffled:
        if r.utility < MIN_UTILITY:
            continue   # skip low-utility targets
        closure = compute_closure(r.id, index)
        new_hours = _hours(closure - selection, index)
        if _hours(selection, index) + new_hours <= budget:
            selection |= closure
    return selection


def _get_neighbors(selection: set, resources: list,
                   index: dict, budget: int) -> list:
    """
    Generate all neighbors of the current solution.

    ADD move:  include a new resource (plus closure) if it fits.
    REMOVE move: drop a leaf (no dependents in selection) from the path.
    SWAP move:  remove a leaf AND add a different resource in one step.
                This is the critical move for escaping local optima — it
                frees the leaf's hours and immediately reinvests them.
                Connection to CSP lecture: "reasignar valores a variables"
                (slide Algoritmos Iterativos para CSP).

    Without SWAP moves, HC finds local optima on the first iteration
    because ADD/REMOVE alone cannot make lateral moves across the space.
    """
    neighbors = []
    remaining = budget - _hours(selection, index)

    # Identify leaf nodes once (used by REMOVE and SWAP moves)
    leaf_ids = [
        rid for rid in selection
        if not any(rid in index[other].prerequisites for other in selection)
    ]

    # ADD moves — add a resource (plus its closure) if it fits and meets threshold
    for r in resources:
        if r.id not in selection and r.utility >= MIN_UTILITY:
            closure = compute_closure(r.id, index)
            added_hours = _hours(closure - selection, index)
            if added_hours <= remaining:
                neighbors.append(selection | closure)

    # REMOVE moves — drop a leaf from the selection
    for rid in leaf_ids:
        neighbors.append(selection - {rid})

    # SWAP moves — replace a leaf with a different resource.
    # This is the key move that allows HC to escape local optima:
    # freeing a leaf's hours creates room for a resource that didn't fit before.
    # Connection to CSP lecture: "reasignar valores a variables" (slide Algoritmos Iterativos).
    for rid_out in leaf_ids:
        after_remove = selection - {rid_out}
        freed = index[rid_out].duration_hours
        new_remaining = remaining + freed

        for r in resources:
            if r.id not in selection and r.utility >= MIN_UTILITY:
                closure = compute_closure(r.id, index)
                added_hours = _hours(closure - after_remove, index)
                if 0 < added_hours <= new_remaining:
                    neighbors.append(after_remove | closure)

    return neighbors


def hill_climbing_solver(resources: list, budget: int,
                         max_iterations: int = 200,
                         restarts: int = 5,
                         seed: int = 42) -> Result:
    """
    Hill climbing with random restarts.

    Each restart:
      1. Generate a random valid solution.
      2. Repeatedly move to the best neighbor (higher utility).
      3. Stop when no neighbor improves utility (local optimum).

    Multiple restarts escape local optima by exploring different regions —
    the exploration/exploitation balance from the metaheuristics lecture.

    Connection to CSP lecture (slide "Algoritmos Iterativos para CSP"):
        "seleccionar variable en conflicto, escoger valor con menos violaciones"
        Here: select the neighbor with highest utility gain.
    """
    index = build_index(resources)
    rng = random.Random(seed)

    best_selection: set = set()
    best_utility: float = 0.0
    total_iters = 0

    for _ in range(restarts):
        current = _random_valid_solution(resources, index, budget, rng)
        current_utility = _utility(current, index)

        for _ in range(max_iterations):
            total_iters += 1
            neighbors = _get_neighbors(current, resources, index, budget)

            if not neighbors:
                break

            best_neighbor = max(neighbors, key=lambda s: _utility(s, index))
            best_neighbor_utility = _utility(best_neighbor, index)

            if best_neighbor_utility > current_utility:
                current = best_neighbor
                current_utility = best_neighbor_utility
            else:
                break   # local optimum

        if current_utility > best_utility:
            best_selection = current
            best_utility = current_utility

    ordered = topological_sort(best_selection, index)
    return Result(
        ordered_path=ordered,
        total_utility=best_utility,
        total_hours=_hours(best_selection, index),
        algorithm="Hill Climbing",
        iterations=total_iters,
        restarts=restarts,
    )


# ── Smoke test (no LLM needed) ─────────────────────────────────────────────────

if __name__ == "__main__":
    with open("resources.json", encoding="utf-8") as f:
        raw = json.load(f)

    dummy_scores = {r["id"]: 5.0 for r in raw}
    resources = load_resources("resources.json", dummy_scores)

    for budget in [15, 20, 30]:
        print(f"\n{'='*55}\nBudget: {budget}h\n{'='*55}")
        print(greedy_solver(resources, budget).summary())
        print()
        print(hill_climbing_solver(resources, budget).summary())
