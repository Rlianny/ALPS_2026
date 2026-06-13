"""
LLM scoring module: uses Groq's llama-3.1-8b-instant to evaluate the utility
of each learning resource given a specific learning goal.

Role of the LLM in the system:
    Bridges natural language (goal + description) and numerical optimization
    (utility coefficient in the objective function). Without LLM scoring, all
    resources would have equal weight and the route would be generic regardless
    of the user's goal.
"""

from __future__ import annotations

import json
import time
import re
import os
import hashlib

# NOTE: `groq` is imported lazily inside the functions that actually call the
# API. This keeps cached / offline runs working even when the package is not
# installed, and means the GROQ_API_KEY is only needed when (re)scoring.

MODEL = "llama-3.1-8b-instant"


def cache_path_for_goal(goal: str, base_dir: str = ".") -> str:
    """
    Map a learning goal to a stable per-goal cache filename.

    Each distinct goal gets its own scores file (e.g. scores_3f9a1c2b4d.json),
    so switching between goals never overwrites another goal's cache. A goal is
    therefore scored by the LLM only once, ever — afterwards every run for that
    goal reads from disk and needs neither the API key nor the groq package.
    """
    digest = hashlib.sha1(goal.strip().encode("utf-8")).hexdigest()[:10]
    return os.path.join(base_dir, f"scores_{digest}.json")


def score_resource(client: Groq, goal: str, resource: dict) -> float:
    """
    Ask the LLM to rate the utility of a resource for a given learning goal.

    Args:
        client:   Groq API client
        goal:     user's learning goal in natural language
        resource: resource dictionary with 'name', 'description', 'duration_hours'

    Returns:
        float in [0.0, 10.0]  (0 = irrelevant, 10 = essential for the goal)
    """
    prompt = f"""You are an expert in AI/ML curriculum design and personalized learning.

User's learning goal:
"{goal}"

Resource to evaluate:
- Name: {resource['name']}
- Description: {resource['description']}
- Duration: {resource['duration_hours']} hours

Rate how useful this resource is for reaching the user's goal.
Consider its direct relevance to the goal — resources that do not directly
address the goal should score lower, even if they are generally useful.

Reply with ONLY a decimal number between 0.0 and 10.0. No other text.
Valid response examples: 8.5 | 3.0 | 9.5 | 2.0"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.1,
    )

    raw = completion.choices[0].message.content.strip()

    # Parse to float with fallback extraction
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        numbers = re.findall(r"\d+\.?\d*", raw)
        if numbers:
            value = float(numbers[0])
            return max(0.0, min(10.0, value))
        print(f"  WARNING: could not parse '{raw}' for '{resource['id']}'. Using 5.0")
        return 5.0


def score_all(goal: str, resources: list, pause: float = 0.5) -> dict:
    """
    Score all resources and return a utility dictionary.

    The pause between calls is necessary because Groq's free tier enforces
    rate limits (requests per minute). Without it, rapid sequential calls
    would trigger HTTP 429 errors and scoring would fail mid-way.

    Args:
        goal:      learning goal in natural language
        resources: list of resource dicts from the dataset
        pause:     seconds to wait between API calls (avoids rate limiting)

    Returns:
        dict mapping resource ID to utility score: {"python_basics": 7.5, ...}
    """
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    utilities = {}

    print(f"Scoring {len(resources)} resources for goal:")
    print(f'  "{goal}"\n')

    for i, resource in enumerate(resources):
        score = score_resource(client, goal, resource)
        utilities[resource["id"]] = score
        print(f"  [{i+1:02d}/{len(resources)}] {resource['name']:<42} → {score:.1f}/10")
        time.sleep(pause)

    avg = sum(utilities.values()) / len(utilities)
    print(f"\nScoring complete. Average utility: {avg:.2f}/10")
    return utilities


def score_all_yielding(goal: str, resources: list, pause: float = 0.5):
    """
    Generator version of score_all — yields a progress dict after each resource is scored.
    Used by the web UI to stream live progress via SSE.
    """
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    for i, resource in enumerate(resources):
        score = score_resource(client, goal, resource)
        yield {
            "index": i + 1,
            "total": len(resources),
            "id": resource["id"],
            "name": resource["name"],
            "score": score,
        }
        time.sleep(pause)


def save_scores(utilities: dict, goal: str, path: str = "scores.json"):
    """Persist scores to disk to avoid re-scoring on every test run."""
    data = {"goal": goal, "utilities": utilities}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Scores saved to '{path}'")


def load_scores(path: str = "scores.json") -> tuple[str, dict]:
    """Load previously computed scores from disk."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["goal"], data["utilities"]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("resources.json", encoding="utf-8") as f:
        resources = json.load(f)

    # Two different goals to verify that scores actually differ between them
    test_goals = [
        "I want to work as an ML engineer in industry, building and deploying models at scale",
        "I want to research NLP and understand how large language models work internally",
    ]

    for goal in test_goals:
        print("=" * 65)
        utilities = score_all(goal, resources)
        filename = "scores_" + goal[:25].replace(" ", "_") + ".json"
        save_scores(utilities, goal, filename)
        print()
