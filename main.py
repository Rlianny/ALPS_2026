"""
main.py — Entry point for the ALPS learning path system.

Pipeline:
  1. Load the resource dataset (resources.json)
  2. Score resources with the LLM for a given learning goal
     (loads from scores.json cache if available; use --rescore to force refresh)
  3. Run both algorithms (Greedy and Hill Climbing) for multiple time budgets
  4. Display a side-by-side comparison of results

Usage:
  python main.py              # uses cached scores if scores.json exists
  python main.py --rescore    # forces the LLM to re-score all resources
"""

import sys
import os
import json

# Load GROQ_API_KEY (and anything else) from a local .env if present, so the
# key never has to be exported by hand. Optional: skipped if python-dotenv
# isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_scorer import score_all, save_scores, load_scores, cache_path_for_goal
from solver import load_resources, build_index, apply_dependency_boost, greedy_solver, hill_climbing_solver, monte_carlo_analysis

# ── Experiment configuration ───────────────────────────────────────────────────

DATASET_FILE = "resources.json"

# The learning goal used for LLM scoring.
# Change this to test different user profiles.
GOAL = "I want to work as an ML engineer building and deploying models in production"

# Time budgets to test — simulates users with different availability
BUDGETS = [15, 20, 30]   # hours

# Hill climbing hyperparameters
HC_CONFIG = {
    "max_iterations": 300,
    "restarts": 5,
    "seed": 42,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def separator(char: str = "─", width: int = 62) -> str:
    return char * width


def print_comparison(budget: int, result_greedy, result_hc):
    """Print a side-by-side comparison of both algorithm results."""
    print(f"\n{'━' * 62}")
    print(f"  Budget: {budget}h")
    print(f"{'━' * 62}")

    # Header row
    print(f"  {'Metric':<22} {'Greedy':>16} {'Hill Climbing':>16}")
    print(f"  {separator()}")

    metrics = [
        ("Total hours",    f"{result_greedy.total_hours}h",
                           f"{result_hc.total_hours}h"),
        ("Total utility",  f"{result_greedy.total_utility:.2f}",
                           f"{result_hc.total_utility:.2f}"),
        ("Resources",      str(len(result_greedy.ordered_path)),
                           str(len(result_hc.ordered_path))),
        ("Iterations",     "—",
                           str(result_hc.iterations)),
    ]

    for label, g_val, hc_val in metrics:
        print(f"  {label:<22} {g_val:>16} {hc_val:>16}")

    # Winner annotation
    if result_hc.total_utility > result_greedy.total_utility:
        winner = "Hill Climbing wins ▲"
        diff = result_hc.total_utility - result_greedy.total_utility
    elif result_greedy.total_utility > result_hc.total_utility:
        winner = "Greedy wins ▲"
        diff = result_greedy.total_utility - result_hc.total_utility
    else:
        winner = "Tie"
        diff = 0.0

    print(f"\n  → {winner}  (Δ utility = {diff:.2f})")

    # Best path (from whichever algorithm won, or greedy on tie)
    best = result_hc if result_hc.total_utility >= result_greedy.total_utility else result_greedy
    print(f"\n  Best path ({best.algorithm}):")
    for i, r in enumerate(best.ordered_path, 1):
        print(f"    {i}. [{r.duration_hours}h | {r.utility:.1f}/10] {r.name}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    force_rescore = "--rescore" in sys.argv

    # ── Step 1: load dataset ───────────────────────────────────────────────────
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: '{DATASET_FILE}' not found. Run generate_dataset.py first.")
        sys.exit(1)

    with open(DATASET_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} resources from '{DATASET_FILE}'.")

    # ── Step 2: LLM scoring ────────────────────────────────────────────────────
    # Each goal has its own cache file, so this goal is scored at most once
    # ever; subsequent runs need neither the API key nor the groq package.
    scores_file = cache_path_for_goal(GOAL)
    if os.path.exists(scores_file) and not force_rescore:
        _, utilities = load_scores(scores_file)
        print(f"Loaded cached scores from '{scores_file}'.")
    else:
        print(f"\nScoring resources with LLM ({len(raw)} API calls)...")
        utilities = score_all(GOAL, raw)
        save_scores(utilities, GOAL, scores_file)

    # ── Step 3: load resources and apply dependency boost ─────────────────────
    resources = load_resources(DATASET_FILE, utilities)
    index = build_index(resources)
    apply_dependency_boost(resources, index, alpha=0.3)

    # Show score table: raw LLM score vs boosted score
    print(f"\n{'─' * 62}")
    print(f"  {'Resource':<42} {'Raw':>5} {'Boosted':>8}")
    print(f"{'─' * 62}")
    for r in sorted(resources, key=lambda x: -x.utility):
        raw = utilities.get(r.id, 5.0)
        delta = f"+{r.utility - raw:.1f}" if r.utility > raw else "  —"
        print(f"  {r.name:<42} {raw:>5.1f} {r.utility:>6.1f}  {delta}")
    print(f"{'─' * 62}")

    # ── Step 4: run experiments ────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  ALPS — AI Learning Path Selector")
    print(f"  Goal: {GOAL[:55]}...")
    print(f"{'═' * 62}")

    for budget in BUDGETS:
        result_greedy = greedy_solver(resources, budget)
        result_hc     = hill_climbing_solver(resources, budget, **HC_CONFIG)
        print_comparison(budget, result_greedy, result_hc)

    # ── Step 5: Monte Carlo analysis of Hill Climbing ──────────────────────
    # HC is non-deterministic: different seeds produce different utilities.
    # Running 30 independent replications gives a statistical characterization
    # of HC performance and validates whether single-run results are representative.
    print(f"\n{'═' * 62}")
    print("  MONTE CARLO ANALYSIS — Hill Climbing (N=30 runs per budget)")
    print(f"  {'Budget':<10} {'Mean':>8} {'Std':>7} {'Min':>8} {'Max':>8} {'95% CI':>20}")
    print(f"  {'─'*60}")

    for budget in BUDGETS:
        mc = monte_carlo_analysis(resources, budget, n_runs=30)
        ci = f"[{mc.ci_lower:.2f}, {mc.ci_upper:.2f}]"
        print(f"  {str(budget)+'h':<10} {mc.mean:>8.2f} {mc.std:>7.2f} "
              f"{mc.minimum:>8.2f} {mc.maximum:>8.2f} {ci:>20}")

    print(f"\n{'═' * 62}")
    print(f"  Done. Scores cached in {scores_file}.")
    print("  Run with --rescore to test a different learning goal.")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
