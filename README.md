# ALPS — AI Learning Path Selector

ALPS builds a personalized learning path from a catalog of educational
resources with prerequisite dependencies. Given a learning goal in natural
language and a time budget in hours, it selects the optimal subset of
resources and orders them into a valid study sequence that maximizes utility
without exceeding the budget.

It combines a **large language model** (semantic relevance scoring) with
**classical combinatorial optimization** (constrained selection + topological
ordering). The LLM only scores; all optimization is done by deterministic
algorithms.

> Final project for *Artificial Intelligence 2025–2026*, Faculty of
> Mathematics and Computer Science, University of Havana.

## Features

- **LLM scoring** of each resource's utility for a given goal
  (`llama-3.1-8b-instant` via the Groq API).
- **Dependency boost** that corrects the LLM's tendency to undervalue
  foundational resources.
- Three solvers for the precedence-constrained knapsack problem:
  - **Greedy** — fast deterministic baseline (utility-per-hour ratio).
  - **Hill Climbing** with random restarts and ADD/REMOVE/**SWAP** moves.
  - **Exact optimum** — branch and bound, used as ground truth to
    measure each heuristic's optimality gap.
- **Topological ordering** (Kahn's algorithm) of the selected resources.
- **Monte Carlo analysis** (N=30) characterizing the Hill Climbing
  distribution with a 95% confidence interval.
- **Per-goal score cache** — each goal is scored once and reused; runs are
  fully offline afterwards.
- **Web UI** (FastAPI + SSE) with live scoring progress and a dark mode.

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) — only needed the
  first time a new goal is scored.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and set GROQ_API_KEY=gsk_...
```

The key is loaded automatically from `.env` (via `python-dotenv`). Cached
goals run without the key and without the `groq` package installed.

## Usage

### Command line

```bash
# Run both heuristics + exact optimum + Monte Carlo for several budgets
python main.py

# Force the LLM to re-score the current goal
python main.py --rescore
```

Edit the `GOAL` and `BUDGETS` variables in [main.py](main.py) to test
different user profiles.

### Web app

```bash
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

Enter a learning objective, pick a budget, and the app scores the resources
(streaming progress live) and shows the path, the algorithm comparison
against the exact optimum, all scores, and the Monte Carlo stability
analysis.

### Regenerating the dataset

```bash
python generate_dataset.py   # validates (no cycles, no broken refs) and writes resources.json
```

### Solver smoke test (no LLM)

```bash
python solver.py             # runs all solvers on dummy scores
```

## How it works

1. **Scoring** — for each resource the LLM receives the goal, the resource
   name/description and duration, and returns a utility score in `[0, 10]`.
   Scores are cached per goal in `scores_<hash>.json`.
2. **Dependency boost** — each resource's score is raised based on the most
   valuable resource that transitively depends on it, so foundational
   prerequisites are not undervalued.
3. **Selection** — Greedy and Hill Climbing select a dependency-closed subset
   that fits the budget and maximizes total (boosted) utility; the exact
   solver computes the provable optimum for comparison.
4. **Ordering** — the selection is sorted topologically so every prerequisite
   precedes the resources that need it.

## Project structure

| File | Purpose |
| ---- | ------- |
| [generate_dataset.py](generate_dataset.py) | Builds and validates the 21-resource dataset (`resources.json`) |
| [llm_scorer.py](llm_scorer.py) | LLM scoring, per-goal caching |
| [solver.py](solver.py) | Greedy, Hill Climbing, exact optimum, topological sort, Monte Carlo |
| [main.py](main.py) | CLI pipeline and side-by-side comparison |
| [app.py](app.py) | FastAPI server (SSE streaming) |
| [index.html](index.html) | Single-page web UI |
| [Informe/Informe.tex](Informe/Informe.tex) | Full project report (Spanish, LaTeX) |

## Reproducibility

LLM scores are not reproducible across runs (even at low temperature), so the
score caches for both goals are committed to the repository and every figure
in the report is regenerated from them. The committed `scores_<hash>.json`
files reproduce all reported results without calling the API.

## Report

The full write-up — problem formulation, formal model, methodology, results
and analysis — is in [Informe/Informe.pdf](Informe/Informe.pdf).

## License

[MIT](LICENSE) © 2026 Lianny Revé Valdivieso
