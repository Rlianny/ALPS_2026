"""
app.py — FastAPI web server for the ALPS learning path UI.

Endpoints:
  GET /                     Serve the single-page UI (index.html)
  GET /api/cache-status     Check whether scores.json exists and matches an objective
  GET /api/run              SSE stream: score resources (if needed) then run both solvers
"""

import asyncio
import json
import os
from pathlib import Path

# Load GROQ_API_KEY from a local .env if present (optional dependency).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from llm_scorer import (
    load_scores,
    save_scores,
    score_resource,
    score_all_yielding,
    cache_path_for_goal,
)
from solver import (
    apply_dependency_boost,
    build_index,
    greedy_solver,
    hill_climbing_solver,
    load_resources,
    monte_carlo_analysis,
)
# NOTE: `groq` is imported lazily inside the request handler so the server can
# start and serve cached results without the package or an API key installed.

BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "resources.json"
INDEX_HTML = BASE_DIR / "index.html"

HC_CONFIG = {"max_iterations": 300, "restarts": 5, "seed": 42}

TAGS: dict[str, tuple[str, str]] = {
    "python_basics":      ("Foundations",   "info"),
    "linear_algebra":     ("Math",          "warning"),
    "calculus":           ("Math",          "warning"),
    "probability":        ("Math",          "warning"),
    "discrete_math":      ("Math",          "warning"),
    "numpy_pandas":       ("Tools",         "info"),
    "data_visualization": ("Tools",         "info"),
    "classical_ml":       ("Core ML",       "success"),
    "optimization":       ("Math",          "warning"),
    "text_preprocessing": ("NLP",           "info"),
    "neural_networks":    ("Deep learning", "info"),
    "model_evaluation":   ("Evaluation",    "warning"),
    "embeddings":         ("NLP",           "info"),
    "cnn":                ("Deep learning", "info"),
    "rnn":                ("Deep learning", "info"),
    "transformers":       ("Deep learning", "info"),
    "llm_fine_tuning":    ("LLMs",          "success"),
    "classical_nlp":      ("NLP",           "info"),
    "basic_rl":           ("RL",            "warning"),
    "mlops":              ("Production",    "danger"),
    "interpretability":   ("Evaluation",    "warning"),
}

app = FastAPI()


def _serialize_path(result, raw_scores: dict) -> list[dict]:
    out = []
    for r in result.ordered_path:
        tag, tc = TAGS.get(r.id, ("Other", "info"))
        out.append({
            "id": r.id,
            "name": r.name,
            "hours": r.duration_hours,
            "score": round(r.utility, 2),
            "raw": round(raw_scores.get(r.id, r.utility), 2),
            "tag": tag,
            "tc": tc,
        })
    return out


def _build_result_payload(budget: int, objective: str, utilities: dict,
                          resources, result_greedy, result_hc) -> dict:
    hc_ids  = {r.id for r in result_hc.ordered_path}
    gr_ids  = {r.id for r in result_greedy.ordered_path}

    all_rows = []
    for r in sorted(resources, key=lambda x: -x.utility):
        all_rows.append({
            "id":      r.id,
            "name":    r.name,
            "hours":   r.duration_hours,
            "raw":     round(utilities.get(r.id, r.utility), 2),
            "boosted": round(r.utility, 2),
            "inH":     r.id in hc_ids,
            "inG":     r.id in gr_ids,
        })

    return {
        "type":      "result",
        "budget":    budget,
        "objective": objective,
        "hc": {
            "total_utility": round(result_hc.total_utility, 2),
            "total_hours":   result_hc.total_hours,
            "algorithm":     result_hc.algorithm,
            "iterations":    result_hc.iterations,
            "restarts":      result_hc.restarts,
            "resources":     _serialize_path(result_hc, utilities),
        },
        "greedy": {
            "total_utility": round(result_greedy.total_utility, 2),
            "total_hours":   result_greedy.total_hours,
            "algorithm":     result_greedy.algorithm,
            "iterations":    0,
            "restarts":      0,
            "resources":     _serialize_path(result_greedy, utilities),
        },
        "all": all_rows,
    }


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/cache-status")
def cache_status(objective: str = Query(default="")):
    scores_path = Path(cache_path_for_goal(objective, str(BASE_DIR)))
    if not scores_path.exists():
        return {"cached": False, "matches": False, "cached_goal": None}
    try:
        cached_goal, _ = load_scores(str(scores_path))
    except Exception:
        return {"cached": False, "matches": False, "cached_goal": None}
    # The file is keyed by goal, so its mere existence means it matches.
    return {"cached": True, "matches": True, "cached_goal": cached_goal}


@app.get("/api/run")
async def run(
    objective: str = Query(...),
    budget: int = Query(...),
    rescore: bool = Query(default=False),
):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not set.")

    if not DATASET_FILE.exists():
        raise HTTPException(status_code=500, detail="resources.json not found.")

    with open(DATASET_FILE, encoding="utf-8") as f:
        raw_resources = json.load(f)

    scores_path = Path(cache_path_for_goal(objective, str(BASE_DIR)))

    async def generate():
        utilities: dict[str, float] = {}

        use_cache = not rescore and scores_path.exists()

        if use_cache:
            _, utilities = load_scores(str(scores_path))
            yield f"data: {json.dumps({'type': 'cached', 'total': len(raw_resources)})}\n\n"
        else:
            from groq import Groq

            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            for i, resource in enumerate(raw_resources):
                score = await asyncio.to_thread(score_resource, client, objective, resource)
                utilities[resource["id"]] = score
                event = {
                    "type":  "scoring",
                    "index": i + 1,
                    "total": len(raw_resources),
                    "name":  resource["name"],
                    "score": round(score, 2),
                }
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.5)

            save_scores(utilities, objective, str(scores_path))

        resources = load_resources(str(DATASET_FILE), utilities)
        index = build_index(resources)
        apply_dependency_boost(resources, index, alpha=0.3)

        result_greedy = await asyncio.to_thread(greedy_solver, resources, budget)
        result_hc     = await asyncio.to_thread(hill_climbing_solver, resources, budget, **HC_CONFIG)
        mc            = await asyncio.to_thread(monte_carlo_analysis, resources, budget, 30)

        payload = _build_result_payload(budget, objective, utilities, resources, result_greedy, result_hc)
        payload["mc"] = {
            "n_runs":  mc.n_runs,
            "mean":    round(mc.mean, 2),
            "std":     round(mc.std, 2),
            "minimum": round(mc.minimum, 2),
            "maximum": round(mc.maximum, 2),
            "ci_lower": round(mc.ci_lower, 2),
            "ci_upper": round(mc.ci_upper, 2),
        }
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
