#!/usr/bin/env python3
"""
batch_tester.py — Batch evaluation harness for SynthesizAIr.

Runs a JSON file of test prompts through one or more model/role combinations,
scores each synthesis with a judge model, and writes results to CSV.

Usage:
    python batch_tester.py test_prompts.json
    python batch_tester.py test_prompts.json -o results.csv -c 2
    python batch_tester.py test_prompts.json --no-judge

JSON input format:
    {
      "prompts": [
        {"id": "p1", "prompt": "...", "category": "Strategy/Planning"}
      ],
      "combinations": [
        {"name": "Default", "use_defaults": true},
        {
          "name": "Custom",
          "sub_models": [{"id": "...", "label": "...", "role": "Analytical"}, ...],
          "master_model": {"id": "...", "label": "..."}
        }
      ],
      "judge_model": {"id": "...", "label": "..."}
    }

If "combinations" is omitted, the default 5 sub-models + master are used.
If "judge_model" is omitted, the default master model is used as judge.
"""

import argparse
import asyncio
import csv
import dataclasses
import datetime
import itertools
import json
import os
import random
import re
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from config import (
    CATEGORIES,
    CATEGORY_NAMES,
    DEFAULT_MASTER_MODEL,
    DEFAULT_NONFREE_MASTER_MODEL,
    DEFAULT_NONFREE_SUB_MODELS,
    DEFAULT_SUB_MODELS,
    DISAGREEMENT_ABSENT_MARKER,
    MAX_TOKENS_DISAGREEMENT,
    MAX_TOKENS_MASTER,
    MAX_TOKENS_SUB,
    OPENROUTER_CHAT_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    ROLE_NAMES,
)
from orchestrator import run_synthesis
from synthesizer import fetch_model_pricing

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Scoring dimensions (order is preserved in CSV columns)
# ---------------------------------------------------------------------------

SCORE_DIMS = [
    "completeness",
    "coherence",
    "diversity_utilized",
    "role_effectiveness",
    "actionability",
    "consensus_vs_tension",
]

# Short header labels for the terminal table
DIM_HEADERS = {
    "completeness":        "Comp",
    "coherence":           "Cohr",
    "diversity_utilized":  "Div",
    "role_effectiveness":  "Role",
    "actionability":       "Act",
    "consensus_vs_tension": "C/T",
}

JUDGE_MAX_TOKENS = 900
JUDGE_TEMPERATURE = 0.1  # low for repeatable scores

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial quality judge evaluating an AI ensemble synthesis. \
Score the synthesis on 6 dimensions, each on a 1–5 scale.

QUESTION: {question}
CATEGORY: {category}

SUB-MODEL RESPONSES (the individual perspectives fed into the synthesis):
{sub_responses}

DISAGREEMENTS DETECTED BEFORE SYNTHESIS:
{disagreements}

FINAL SYNTHESIS (the output to evaluate):
{synthesis}

SCORING RUBRIC:
- completeness (1–5): Does the synthesis cover all relevant aspects?
  5=comprehensive, no gaps | 1=major aspects unaddressed
- coherence (1–5): Is the synthesis well-structured and internally consistent?
  5=exemplary structure and flow | 1=disorganised or self-contradictory
- diversity_utilized (1–5): Did the synthesis draw meaningfully from all cognitive roles?
  5=all roles visibly contributed distinct value | 1=indistinguishable from single-model
- role_effectiveness (1–5): Did each role (Analytical, Devil's Advocate, Creative, \
Pragmatist, Synthesizer) contribute in line with its defined purpose?
  5=each role's unique value is clearly evident | 1=roles were redundant or ignored
- actionability (1–5): Does the synthesis provide clear, usable guidance or conclusions?
  5=immediately actionable with concrete steps | 1=purely abstract or vague
- consensus_vs_tension (1–5): Were sub-model disagreements detected, surfaced, and resolved?
  5=tensions expertly navigated | 3=models agreed (neutral) | 1=real tensions ignored

Respond with ONLY the following JSON object — no markdown fences, no commentary:
{{
  "completeness":         {{"score": N, "reasoning": "one sentence"}},
  "coherence":            {{"score": N, "reasoning": "one sentence"}},
  "diversity_utilized":   {{"score": N, "reasoning": "one sentence"}},
  "role_effectiveness":   {{"score": N, "reasoning": "one sentence"}},
  "actionability":        {{"score": N, "reasoning": "one sentence"}},
  "consensus_vs_tension": {{"score": N, "reasoning": "one sentence"}},
  "overall_reasoning":    "2–3 sentence overall assessment"
}}\
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TestPrompt:
    id: str
    prompt: str
    category: str


@dataclasses.dataclass
class Combination:
    name: str
    sub_models: list[dict]
    master_model: dict


@dataclasses.dataclass
class DimScore:
    score: int      # 1–5; 0 means unavailable
    reasoning: str


@dataclasses.dataclass
class ScoreResult:
    completeness: DimScore
    coherence: DimScore
    diversity_utilized: DimScore
    role_effectiveness: DimScore
    actionability: DimScore
    consensus_vs_tension: DimScore
    overall_reasoning: str
    judge_failed: bool = False

    @property
    def total(self) -> int:
        return sum(getattr(self, d).score for d in SCORE_DIMS)


@dataclasses.dataclass
class RunResult:
    run_id: str
    prompt: TestPrompt
    combination: Combination
    synthesis_content: str | None
    disagreements_content: str | None
    sub_results: list
    scores: ScoreResult | None
    elapsed_seconds: float
    error: str | None
    combination_id: str = ""
    phase: str = ""


# ---------------------------------------------------------------------------
# Loading the JSON test file
# ---------------------------------------------------------------------------


def _combo_from_dict(raw: dict) -> Combination:
    if raw.get("use_defaults"):
        return Combination(
            raw.get("name", "Default"),
            list(DEFAULT_SUB_MODELS),
            dict(DEFAULT_MASTER_MODEL),
        )
    return Combination(
        raw.get("name", "Unnamed"),
        raw.get("sub_models", list(DEFAULT_SUB_MODELS)),
        raw.get("master_model", dict(DEFAULT_MASTER_MODEL)),
    )


def load_test_file(path: str) -> tuple[list[TestPrompt], list[Combination], dict]:
    """Parse the JSON test file. Returns (prompts, combinations, judge_model)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    prompts = [
        TestPrompt(p["id"], p["prompt"], p["category"])
        for p in data["prompts"]
    ]

    raw_combos = data.get("combinations", [{"name": "Default", "use_defaults": True}])
    combinations = [_combo_from_dict(c) for c in raw_combos]

    judge_model = data.get("judge_model", dict(DEFAULT_MASTER_MODEL))
    return prompts, combinations, judge_model


# ---------------------------------------------------------------------------
# Combination matrix generator
# ---------------------------------------------------------------------------


def generate_combinations(
    model_pool: list[dict],
    role_pool: list[str] | None = None,
    phases: list[int] | None = None,
    max_combinations: int = 10,
    master_model: dict | None = None,
) -> list[dict]:
    """
    Generate test matrix combinations across three experiment phases.

    Returns list of dicts with keys:
        combination: Combination, combination_id: str, phase: int, phase_label: str
    """
    if role_pool is None:
        role_pool = list(ROLE_NAMES)
    if phases is None:
        phases = [1, 2, 3]
    if master_model is None:
        master_model = dict(DEFAULT_MASTER_MODEL)

    entries: list[dict] = []

    # Phase 1: Same model × 5 slots, all different roles (role isolation test)
    if 1 in phases:
        p1: list[Combination] = []
        for model in model_pool:
            sub_models = [
                {"id": model["id"], "label": model["label"], "role": role}
                for role in role_pool[:5]
            ]
            p1.append(Combination(
                name=f"P1-{model['label'][:20]}",
                sub_models=sub_models,
                master_model=dict(master_model),
            ))
        if len(p1) > max_combinations:
            p1 = random.sample(p1, max_combinations)
        for i, c in enumerate(p1):
            entries.append({
                "combination": c,
                "combination_id": f"p1_{i + 1:03d}",
                "phase": 1,
                "phase_label": "Phase 1: Role Isolation",
            })

    # Phase 2: Different models × same role across all slots (model diversity test)
    if 2 in phases:
        if len(model_pool) < 5:
            console.print("[yellow]Warning: Phase 2 requires ≥5 models in pool. Skipping.[/]")
        else:
            p2: list[Combination] = []
            for role in role_pool[:5]:
                for mg in itertools.combinations(model_pool, 5):
                    sub_models = [
                        {"id": m["id"], "label": m["label"], "role": role}
                        for m in mg
                    ]
                    p2.append(Combination(
                        name=f"P2-{role[:14]}",
                        sub_models=sub_models,
                        master_model=dict(master_model),
                    ))
            if len(p2) > max_combinations:
                p2 = random.sample(p2, max_combinations)
            # Deduplicate names by appending index
            for i, c in enumerate(p2):
                c.name = f"P2-{c.name.split('-', 1)[1]}-{i + 1:02d}"
                entries.append({
                    "combination": c,
                    "combination_id": f"p2_{i + 1:03d}",
                    "phase": 2,
                    "phase_label": "Phase 2: Model Diversity",
                })

    # Phase 3: Different models × different roles (full combination test)
    if 3 in phases:
        if len(model_pool) < 5:
            console.print("[yellow]Warning: Phase 3 requires ≥5 models in pool. Skipping.[/]")
        else:
            mg_list = list(itertools.combinations(model_pool, 5))
            rp_list = list(itertools.permutations(role_pool[:5]))
            candidates = list(itertools.product(mg_list, rp_list))
            if len(candidates) > max_combinations:
                candidates = random.sample(candidates, max_combinations)
            p3: list[Combination] = []
            for mg, rp in candidates:
                sub_models = [
                    {"id": mg[j]["id"], "label": mg[j]["label"], "role": rp[j]}
                    for j in range(5)
                ]
                p3.append(Combination(
                    name=f"P3-mix-{len(p3) + 1:02d}",
                    sub_models=sub_models,
                    master_model=dict(master_model),
                ))
            if len(p3) > max_combinations:
                p3 = p3[:max_combinations]
            for i, c in enumerate(p3):
                entries.append({
                    "combination": c,
                    "combination_id": f"p3_{i + 1:03d}",
                    "phase": 3,
                    "phase_label": "Phase 3: Full Combination",
                })

    return entries


def estimate_cost(
    entries: list[dict],
    prompts: list[TestPrompt],
    master_model: dict,
    judge_model: dict | None,
    pricing: dict[str, dict[str, float]],
    avg_prompt_tokens: int = 300,
) -> dict:
    """
    Estimate the cost of a batch run using OpenRouter pricing.

    Assumptions per prompt x combination run:
      - 5 sub-model calls: ~avg_prompt_tokens in, MAX_TOKENS_SUB out each
      - 1 master independent: ~avg_prompt_tokens in, MAX_TOKENS_SUB out
      - 1 disagreement detection: ~(5 * MAX_TOKENS_SUB + avg_prompt_tokens) in, MAX_TOKENS_DISAGREEMENT out
      - 1 synthesis: ~(5 * MAX_TOKENS_SUB + MAX_TOKENS_SUB + avg_prompt_tokens) in, MAX_TOKENS_MASTER out
      - 1 judge call (if enabled): ~(5 * MAX_TOKENS_SUB + MAX_TOKENS_MASTER) in, JUDGE_MAX_TOKENS out

    Returns dict with per-model breakdown and total.
    """
    total_cost = 0.0
    model_costs: dict[str, float] = {}
    unknown_models: set[str] = set()

    def _add(model_id: str, input_tokens: int, output_tokens: int) -> float:
        p = pricing.get(model_id, None)
        if p is None:
            unknown_models.add(model_id)
            return 0.0
        cost = p["prompt"] * input_tokens + p["completion"] * output_tokens
        model_costs[model_id] = model_costs.get(model_id, 0.0) + cost
        return cost

    sub_responses_tokens = 5 * MAX_TOKENS_SUB
    disagree_input = sub_responses_tokens + MAX_TOKENS_SUB + avg_prompt_tokens
    synth_input = sub_responses_tokens + MAX_TOKENS_SUB + avg_prompt_tokens
    judge_input = sub_responses_tokens + MAX_TOKENS_MASTER

    for _ in prompts:
        for e in entries:
            combo = e["combination"]
            # 5 sub-model calls
            for sm in combo.sub_models:
                total_cost += _add(sm["id"], avg_prompt_tokens, MAX_TOKENS_SUB)
            # Master independent view
            master_id = combo.master_model["id"]
            total_cost += _add(master_id, avg_prompt_tokens, MAX_TOKENS_SUB)
            # Disagreement detection (master)
            total_cost += _add(master_id, disagree_input, MAX_TOKENS_DISAGREEMENT)
            # Synthesis (master)
            total_cost += _add(master_id, synth_input, MAX_TOKENS_MASTER)
            # Judge
            if judge_model:
                total_cost += _add(judge_model["id"], judge_input, JUDGE_MAX_TOKENS)

    return {
        "total": total_cost,
        "by_model": model_costs,
        "unknown_models": unknown_models,
    }


def write_matrix_preview(
    entries: list[dict],
    prompts: list[TestPrompt],
    model_pool: list[dict],
    master_model: dict,
    categories: list[str] | None,
    output_path: str = "test_matrix.json",
) -> int:
    """
    Write a preview JSON of the test matrix and return estimated API call count.

    Per prompt × combination:
      5 sub-model calls + 1 master initial + 1 disagreement detection + 1 synthesis + 1 judge = 9
    """
    calls_per_run = 9
    total_runs = len(prompts) * len(entries)
    estimated_calls = total_runs * calls_per_run

    phase_counts: dict[str, int] = {}
    for e in entries:
        label = e["phase_label"]
        phase_counts[label] = phase_counts.get(label, 0) + 1

    preview = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_pool": model_pool,
        "master_model": master_model,
        "categories_filter": categories,
        "prompts_count": len(prompts),
        "phases": phase_counts,
        "total_combinations": len(entries),
        "total_runs": total_runs,
        "estimated_api_calls": estimated_calls,
        "combinations": [
            {
                "combination_id": e["combination_id"],
                "phase": e["phase_label"],
                "name": e["combination"].name,
                "sub_models": e["combination"].sub_models,
                "master_model": e["combination"].master_model,
            }
            for e in entries
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2)

    return estimated_calls


def print_matrix_preview(
    entries: list[dict],
    prompts: list[TestPrompt],
    estimated_calls: int,
    cost_estimate: dict | None = None,
) -> None:
    """Display the matrix preview in the terminal."""
    phase_counts: dict[str, int] = {}
    for e in entries:
        label = e["phase_label"]
        phase_counts[label] = phase_counts.get(label, 0) + 1

    console.print(f"\n[bold cyan]═══ Test Matrix Preview ═══[/]\n")
    for label, count in sorted(phase_counts.items()):
        console.print(f"  {label}: [yellow]{count}[/] combinations")
    console.print(f"\n  Prompts           : [yellow]{len(prompts)}[/]")
    console.print(f"  Total combinations: [yellow]{len(entries)}[/]")
    console.print(f"  Total runs        : [yellow]{len(prompts) * len(entries)}[/]")
    console.print(f"  Estimated API calls: [bold yellow]{estimated_calls}[/]")

    if cost_estimate:
        total = cost_estimate["total"]
        if total > 0:
            console.print(f"  Estimated cost    : [bold yellow]${total:.4f}[/] [dim](worst-case, max output tokens)[/]")
            by_model = cost_estimate["by_model"]
            if by_model:
                console.print(f"\n  [dim]Cost breakdown by model:[/]")
                for model_id, cost in sorted(by_model.items(), key=lambda x: -x[1]):
                    console.print(f"    {model_id:<45} [yellow]${cost:.4f}[/]")
        else:
            console.print(f"  Estimated cost    : [green]$0.00 (free tier)[/]")
        if cost_estimate.get("unknown_models"):
            console.print(
                f"  [dim yellow]Note: pricing unavailable for: "
                f"{', '.join(cost_estimate['unknown_models'])}[/]"
            )

    table = Table(header_style="bold magenta", show_lines=False, show_edge=True)
    table.add_column("ID", style="dim", width=8)
    table.add_column("Phase", max_width=28)
    table.add_column("Name", max_width=22)
    table.add_column("Sub-models", max_width=50)

    for e in entries:
        combo = e["combination"]
        models_str = ", ".join(
            f"{sm['role'][:4]}={sm['label'][:12]}" for sm in combo.sub_models
        )
        table.add_row(
            e["combination_id"],
            e["phase_label"],
            combo.name,
            models_str,
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Judge model call
# ---------------------------------------------------------------------------


def _build_sub_responses_block(sub_results: list) -> str:
    parts = []
    for r in sub_results:
        role_tag = f"[{r.role.upper()}] " if r.role else ""
        if r.content:
            snippet = r.content[:500] + ("…" if len(r.content) > 500 else "")
            parts.append(f"{role_tag}{r.label}:\n{snippet}")
        else:
            parts.append(f"{role_tag}{r.label}: [FAILED — {r.error}]")
    return "\n\n".join(parts)


def _parse_judge_response(content: str) -> ScoreResult:
    """Extract and parse the JSON scoring object from the judge's response."""
    text = content.strip()

    # Strip ```json ... ``` fencing if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # Find outermost { ... }
        bare = re.search(r"\{.*\}", text, re.DOTALL)
        if bare:
            text = bare.group(0)

    def _failed(msg: str) -> ScoreResult:
        fd = DimScore(0, "")
        return ScoreResult(
            completeness=fd, coherence=fd, diversity_utilized=fd,
            role_effectiveness=fd, actionability=fd, consensus_vs_tension=fd,
            overall_reasoning=msg, judge_failed=True,
        )

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return _failed(f"Judge JSON parse error: {exc} — raw: {content[:120]}")

    def _extract(key: str) -> DimScore:
        raw = obj.get(key, {})
        if isinstance(raw, dict):
            score = int(raw.get("score", 0))
            reasoning = str(raw.get("reasoning", ""))
        else:
            score = int(raw) if isinstance(raw, (int, float)) else 0
            reasoning = ""
        return DimScore(max(1, min(5, score)), reasoning)

    return ScoreResult(
        completeness=_extract("completeness"),
        coherence=_extract("coherence"),
        diversity_utilized=_extract("diversity_utilized"),
        role_effectiveness=_extract("role_effectiveness"),
        actionability=_extract("actionability"),
        consensus_vs_tension=_extract("consensus_vs_tension"),
        overall_reasoning=str(obj.get("overall_reasoning", "")),
    )


async def _call_judge(
    client: httpx.AsyncClient,
    judge_model: dict,
    test_prompt: TestPrompt,
    synthesis_content: str,
    sub_results: list,
    disagreements_content: str | None,
    api_key: str,
) -> ScoreResult:
    disagree_block = (
        disagreements_content.strip()
        if (
            disagreements_content
            and not disagreements_content.strip().upper().startswith(DISAGREEMENT_ABSENT_MARKER)
        )
        else "None detected — sub-models broadly agreed."
    )

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=test_prompt.prompt,
        category=test_prompt.category,
        sub_responses=_build_sub_responses_block(sub_results),
        disagreements=disagree_block,
        synthesis=synthesis_content[:2000],
    )

    def _failed(msg: str) -> ScoreResult:
        fd = DimScore(0, "")
        return ScoreResult(
            completeness=fd, coherence=fd, diversity_utilized=fd,
            role_effectiveness=fd, actionability=fd, consensus_vs_tension=fd,
            overall_reasoning=msg, judge_failed=True,
        )

    try:
        response = await client.post(
            OPENROUTER_CHAT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/local/synthesizair",
                "X-Title": "SynthesizAIr",
            },
            json={
                "model": judge_model["id"],
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": JUDGE_MAX_TOKENS,
                "temperature": JUDGE_TEMPERATURE,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_judge_response(content)
    except httpx.HTTPStatusError as exc:
        return _failed(f"Judge HTTP {exc.response.status_code}: {exc.response.text[:120]}")
    except Exception as exc:
        return _failed(f"Judge call failed: {exc}")


# ---------------------------------------------------------------------------
# Single run: synthesis + judge
# ---------------------------------------------------------------------------


async def run_single(
    test_prompt: TestPrompt,
    combination: Combination,
    judge_model: dict | None,
    api_key: str,
    run_id: str,
    combination_id: str = "",
    phase: str = "",
) -> RunResult:
    start = time.monotonic()

    try:
        outcome = await run_synthesis(
            test_prompt.prompt,
            api_key,
            combination.sub_models,
            combination.master_model,
            on_result=None,
            category=test_prompt.category,
        )
        elapsed = time.monotonic() - start

        synthesis_content = outcome["synthesis"].content
        disagree_result = outcome.get("disagreements")
        disagreements_content = disagree_result.content if disagree_result else None
        sub_results = outcome["sub_results"]
        synthesis_error = outcome["synthesis"].error if not synthesis_content else None

        scores: ScoreResult | None = None
        if judge_model and synthesis_content:
            async with httpx.AsyncClient() as judge_client:
                scores = await _call_judge(
                    judge_client, judge_model, test_prompt,
                    synthesis_content, sub_results, disagreements_content, api_key,
                )

        return RunResult(
            run_id=run_id,
            prompt=test_prompt,
            combination=combination,
            synthesis_content=synthesis_content,
            disagreements_content=disagreements_content,
            sub_results=sub_results,
            scores=scores,
            elapsed_seconds=elapsed,
            error=synthesis_error,
            combination_id=combination_id,
            phase=phase,
        )

    except Exception as exc:
        return RunResult(
            run_id=run_id,
            prompt=test_prompt,
            combination=combination,
            synthesis_content=None,
            disagreements_content=None,
            sub_results=[],
            scores=None,
            elapsed_seconds=time.monotonic() - start,
            error=str(exc),
            combination_id=combination_id,
            phase=phase,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_batch(
    prompts: list[TestPrompt],
    combinations: list[Combination],
    judge_model: dict | None,
    api_key: str,
    concurrency: int,
    progress: Progress,
    combo_metadata: dict[str, dict] | None = None,
) -> list[RunResult]:
    """
    Run all prompt × combination pairs.

    combo_metadata: optional mapping from combo.name to
        {"combination_id": str, "phase": str} for matrix runs.
    """
    semaphore = asyncio.Semaphore(concurrency)
    ts_tag = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    total = len(prompts) * len(combinations)
    task_id = progress.add_task("Starting…", total=total)
    meta = combo_metadata or {}

    async def _run_one(p: TestPrompt, combo: Combination) -> RunResult:
        async with semaphore:
            run_id = f"{p.id}__{combo.name}__{ts_tag}"
            cm = meta.get(combo.name, {})
            progress.update(
                task_id,
                description=f"[cyan]{p.id}[/] × [yellow]{combo.name}[/]",
            )
            result = await run_single(
                p, combo, judge_model, api_key, run_id,
                combination_id=cm.get("combination_id", ""),
                phase=cm.get("phase", ""),
            )
            progress.advance(task_id)
            return result

    coros = [
        _run_one(p, combo)
        for p in prompts
        for combo in combinations
    ]
    return list(await asyncio.gather(*coros))


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _compute_summary(results: list[RunResult]) -> list[dict]:
    """
    For each prompt category, find the combination with the highest average
    total_score. Returns a list of summary row dicts ready for CSV output.
    """
    from collections import defaultdict

    # category -> combo_name -> list of total scores
    totals: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    # category -> combo_name -> dim -> list of scores
    dims: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for r in results:
        if r.scores and not r.scores.judge_failed and r.scores.total > 0:
            cat, combo = r.prompt.category, r.combination.name
            totals[cat][combo].append(r.scores.total)
            for d in SCORE_DIMS:
                dims[cat][combo][d].append(getattr(r.scores, d).score)

    rows = []
    for cat in sorted(totals):
        avg_by_combo = {
            combo: sum(ts) / len(ts)
            for combo, ts in totals[cat].items()
        }
        if not avg_by_combo:
            continue
        best = max(avg_by_combo, key=avg_by_combo.__getitem__)
        best_avg = avg_by_combo[best]
        n_runs = len(totals[cat][best])

        row: dict[str, Any] = {
            "run_id": "SUMMARY",
            "timestamp": "",
            "prompt_id": cat,
            "prompt_snippet": f"Best combination for: {cat}",
            "category": cat,
            "combination": best,
            "disagreements_found": "",
            "synthesis_length_chars": "",
            "elapsed_seconds": "",
        }
        for d in SCORE_DIMS:
            vals = dims[cat][best].get(d, [])
            row[d] = f"{sum(vals)/len(vals):.2f}" if vals else ""
        row["total_score"] = f"{best_avg:.2f}"
        row["judge_reasoning"] = (
            f"Best avg total {best_avg:.2f}/30 over {n_runs} run(s). "
            + "  |  ".join(
                f"{c}: {avg_by_combo[c]:.2f}"
                for c in sorted(avg_by_combo, key=avg_by_combo.__getitem__, reverse=True)
            )
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "run_id", "combination_id", "phase", "timestamp",
    "prompt_id", "prompt_snippet", "category", "combination",
    "completeness", "coherence", "diversity_utilized",
    "role_effectiveness", "actionability", "consensus_vs_tension",
    "total_score",
    "disagreements_found", "synthesis_length_chars", "elapsed_seconds",
    "judge_reasoning",
]


def _result_to_row(r: RunResult, ts: str) -> dict[str, Any]:
    disagree_present = (
        r.disagreements_content is not None
        and not r.disagreements_content.strip().upper().startswith(DISAGREEMENT_ABSENT_MARKER)
    )
    row: dict[str, Any] = {
        "run_id": r.run_id,
        "combination_id": r.combination_id,
        "phase": r.phase,
        "timestamp": ts,
        "prompt_id": r.prompt.id,
        "prompt_snippet": r.prompt.prompt[:100].replace("\n", " "),
        "category": r.prompt.category,
        "combination": r.combination.name,
        "disagreements_found": "yes" if disagree_present else "no",
        "synthesis_length_chars": len(r.synthesis_content) if r.synthesis_content else 0,
        "elapsed_seconds": f"{r.elapsed_seconds:.1f}",
    }

    if r.error and not r.synthesis_content:
        for d in SCORE_DIMS:
            row[d] = ""
        row["total_score"] = ""
        row["judge_reasoning"] = f"RUN FAILED: {r.error}"
        return row

    if r.scores and not r.scores.judge_failed:
        for d in SCORE_DIMS:
            row[d] = getattr(r.scores, d).score
        row["total_score"] = r.scores.total
        dim_notes = " | ".join(
            f"{d}={getattr(r.scores, d).reasoning}"
            for d in SCORE_DIMS
            if getattr(r.scores, d).reasoning
        )
        row["judge_reasoning"] = (
            r.scores.overall_reasoning
            + ("  ||  " + dim_notes if dim_notes else "")
        )
    else:
        for d in SCORE_DIMS:
            row[d] = ""
        row["total_score"] = ""
        row["judge_reasoning"] = (
            r.scores.overall_reasoning
            if r.scores
            else "Judge disabled or not run"
        )

    return row


def write_csv(results: list[RunResult], output_path: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    data_rows = [_result_to_row(r, ts) for r in results]
    summary_rows = _compute_summary(results)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data_rows)
        if summary_rows:
            writer.writerow({k: "" for k in CSV_FIELDS})  # blank separator
            for row in summary_rows:
                writer.writerow(row)

    console.print(f"\n[bold green]Results written to:[/] {output_path}")


# ---------------------------------------------------------------------------
# Terminal summary table
# ---------------------------------------------------------------------------


def print_terminal_summary(results: list[RunResult]) -> None:
    table = Table(
        title="Batch Results",
        header_style="bold magenta",
        show_lines=False,
        show_edge=True,
    )
    table.add_column("ID", style="dim", width=6)
    table.add_column("Category", max_width=22)
    table.add_column("Combination", max_width=14)
    for dim in SCORE_DIMS:
        table.add_column(DIM_HEADERS[dim], justify="right", width=5)
    table.add_column("Total", justify="right", style="bold cyan", width=6)
    table.add_column("Disagree", justify="center", width=8)
    table.add_column("Time", justify="right", style="dim", width=7)

    for r in results:
        cat_color = CATEGORIES.get(r.prompt.category, {}).get("color", "white")
        cat_cell = f"[{cat_color}]{r.prompt.category[:20]}[/]"
        disagree_present = (
            r.disagreements_content is not None
            and not r.disagreements_content.strip().upper().startswith(
                DISAGREEMENT_ABSENT_MARKER
            )
        )
        disagree_cell = "[orange3]yes[/]" if disagree_present else "[dim]no[/]"

        if r.error and not r.synthesis_content:
            dim_cells = ["—"] * len(SCORE_DIMS)
            total_cell = "[red]ERR[/]"
        elif r.scores and not r.scores.judge_failed:
            dim_cells = [str(getattr(r.scores, d).score) for d in SCORE_DIMS]
            total_cell = str(r.scores.total)
        else:
            dim_cells = ["—"] * len(SCORE_DIMS)
            total_cell = "—"

        table.add_row(
            r.prompt.id,
            cat_cell,
            r.combination.name[:14],
            *dim_cells,
            total_cell,
            disagree_cell,
            f"{r.elapsed_seconds:.1f}s",
        )

    console.print()
    console.print(table)

    summary = _compute_summary(results)
    if summary:
        console.print("\n[bold]Best combination per category:[/]")
        for row in summary:
            cat_color = CATEGORIES.get(row["category"], {}).get("color", "white")
            console.print(
                f"  [{cat_color}]{row['category']}[/]  →  "
                f"[yellow]{row['combination']}[/]  "
                f"(avg total [bold cyan]{row['total_score']}[/]/30)"
            )


# ---------------------------------------------------------------------------
# Matrix experiment summary — phase-specific findings
# ---------------------------------------------------------------------------


def print_matrix_summary(results: list[RunResult]) -> None:
    """Print enhanced summary with per-phase findings for matrix experiments."""
    from collections import defaultdict

    scored = [r for r in results if r.scores and not r.scores.judge_failed and r.scores.total > 0]
    if not scored:
        console.print("\n[yellow]No scored results to summarise.[/]")
        return

    # ── Best combination per category ──
    cat_scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in scored:
        key = r.combination_id or r.combination.name
        cat_scores[r.prompt.category][key].append(r.scores.total)

    console.print(f"\n[bold cyan]═══ Matrix Experiment Summary ═══[/]\n")
    console.print("[bold]Best combination per prompt category:[/]")
    for cat in sorted(cat_scores):
        avgs = {k: sum(v) / len(v) for k, v in cat_scores[cat].items()}
        if not avgs:
            continue
        best = max(avgs, key=avgs.__getitem__)
        cat_color = CATEGORIES.get(cat, {}).get("color", "white")
        console.print(
            f"  [{cat_color}]{cat}[/]  →  [yellow]{best}[/]  "
            f"(avg [bold cyan]{avgs[best]:.1f}[/]/30)"
        )

    # ── Best combination overall ──
    all_scores: dict[str, list[int]] = defaultdict(list)
    for r in scored:
        key = r.combination_id or r.combination.name
        all_scores[key].append(r.scores.total)
    all_avgs = {k: sum(v) / len(v) for k, v in all_scores.items()}
    if all_avgs:
        best_overall = max(all_avgs, key=all_avgs.__getitem__)
        console.print(
            f"\n[bold]Best combination overall:[/]  "
            f"[bold yellow]{best_overall}[/]  (avg [bold cyan]{all_avgs[best_overall]:.1f}[/]/30)"
        )
        best_r = next((r for r in scored if (r.combination_id or r.combination.name) == best_overall), None)
        if best_r:
            for sm in best_r.combination.sub_models:
                console.print(f"    {sm['role']}: {sm['label']}")

    # ── Phase-specific findings ──
    phase_results: dict[str, list[RunResult]] = defaultdict(list)
    for r in scored:
        if r.phase:
            phase_results[r.phase].append(r)

    # Phase 1: Role Isolation
    p1 = phase_results.get("Phase 1: Role Isolation", [])
    if p1:
        console.print(f"\n[bold magenta]── Phase 1: Role Isolation ──[/]")
        avg_role_eff = sum(r.scores.role_effectiveness.score for r in p1) / len(p1)
        avg_div = sum(r.scores.diversity_utilized.score for r in p1) / len(p1)
        meaningful = avg_role_eff >= 3.0 and avg_div >= 3.0
        console.print(f"  Avg role_effectiveness: {avg_role_eff:.1f}/5  |  Avg diversity_utilized: {avg_div:.1f}/5")
        if meaningful:
            console.print("  Finding: [green]Yes[/] — roles produced meaningfully different outputs")
        else:
            console.print("  Finding: [yellow]Limited[/] — role differentiation was weak with same-model setups")
        # Best single-model performer
        p1_avgs: dict[str, list[int]] = defaultdict(list)
        for r in p1:
            p1_avgs[r.combination_id or r.combination.name].append(r.scores.total)
        p1_means = {k: sum(v) / len(v) for k, v in p1_avgs.items()}
        if p1_means:
            best_p1 = max(p1_means, key=p1_means.__getitem__)
            console.print(f"  Best single-model performer: [yellow]{best_p1}[/] (avg {p1_means[best_p1]:.1f}/30)")

    # Phase 2: Model Diversity
    p2 = phase_results.get("Phase 2: Model Diversity", [])
    if p2:
        console.print(f"\n[bold magenta]── Phase 2: Model Diversity ──[/]")
        # Group by combination to find most consistent (lowest variance)
        p2_by_combo: dict[str, list[int]] = defaultdict(list)
        for r in p2:
            p2_by_combo[r.combination_id or r.combination.name].append(r.scores.total)
        p2_means = {k: sum(v) / len(v) for k, v in p2_by_combo.items()}
        # Most consistent = highest average with lowest spread
        if p2_means:
            best_p2 = max(p2_means, key=p2_means.__getitem__)
            console.print(f"  Tested {len(p2_means)} model-diversity combinations")
            console.print(f"  Most consistent performer: [yellow]{best_p2}[/] (avg {p2_means[best_p2]:.1f}/30)")
            # Show which role was used in the best combo
            best_r = next((r for r in p2 if (r.combination_id or r.combination.name) == best_p2), None)
            if best_r:
                role_used = best_r.combination.sub_models[0]["role"]
                console.print(f"  Role assigned: [cyan]{role_used}[/]")

    # Phase 3: Full Combination
    p3 = phase_results.get("Phase 3: Full Combination", [])
    if p3:
        console.print(f"\n[bold magenta]── Phase 3: Full Combination ──[/]")
        p3_by_combo: dict[str, list[int]] = defaultdict(list)
        for r in p3:
            p3_by_combo[r.combination_id or r.combination.name].append(r.scores.total)
        p3_means = {k: sum(v) / len(v) for k, v in p3_by_combo.items()}
        if p3_means:
            best_p3 = max(p3_means, key=p3_means.__getitem__)
            console.print(f"  Winning full combination: [bold yellow]{best_p3}[/] (avg {p3_means[best_p3]:.1f}/30)")
            best_r = next((r for r in p3 if (r.combination_id or r.combination.name) == best_p3), None)
            if best_r:
                for sm in best_r.combination.sub_models:
                    console.print(f"    {sm['role']}: {sm['label']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by both 'run' and 'generate' subcommands."""
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (default: batch_results_TIMESTAMP.csv)",
    )
    parser.add_argument(
        "--judge",
        help="Override judge model ID (e.g. mistralai/mistral-small-3.1-24b-instruct:free)",
        metavar="MODEL_ID",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int, default=1,
        help="Max parallel prompt×combination runs (default: 1, safe for free-tier)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip judge scoring — only synthesis metrics recorded",
    )


def _resolve_judge(args: argparse.Namespace, default_judge: dict) -> dict | None:
    if args.no_judge:
        return None
    if args.judge:
        return {"id": args.judge, "label": args.judge}
    return default_judge


def _run_and_report(
    prompts: list[TestPrompt],
    combinations: list[Combination],
    judge_model: dict | None,
    api_key: str,
    concurrency: int,
    output_path: str,
    combo_metadata: dict[str, dict] | None = None,
    is_matrix: bool = False,
) -> None:
    """Shared execution: run batch, print summary, write CSV."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        results = asyncio.run(
            run_batch(
                prompts, combinations, judge_model,
                api_key, concurrency, progress,
                combo_metadata=combo_metadata,
            )
        )

    print_terminal_summary(results)
    if is_matrix:
        print_matrix_summary(results)
    write_csv(results, output_path)


def cmd_run(args: argparse.Namespace) -> None:
    """Handler for the 'run' subcommand (original batch tester behaviour)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] OPENROUTER_API_KEY not set.")
        sys.exit(1)

    try:
        prompts, combinations, judge_model = load_test_file(args.json_file)
    except FileNotFoundError:
        console.print(f"[bold red]File not found:[/] {args.json_file}")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as exc:
        console.print(f"[bold red]Invalid test file:[/] {exc}")
        sys.exit(1)

    judge_model = _resolve_judge(args, judge_model)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = args.output or f"batch_results_{ts}.csv"

    console.print(f"\n[bold cyan]SynthesizAIr Batch Tester[/]")
    console.print(f"  Prompts      : {len(prompts)}")
    console.print(
        f"  Combinations : {len(combinations)}"
        f"  ({', '.join(c.name for c in combinations)})"
    )
    console.print(f"  Total runs   : {len(prompts) * len(combinations)}")
    console.print(
        f"  Judge model  : "
        + (f"[green]{judge_model['label']}[/]" if judge_model else "[dim]disabled[/]")
    )
    console.print(f"  Concurrency  : {args.concurrency}")
    console.print(f"  Output       : {output_path}\n")

    _run_and_report(prompts, combinations, judge_model, api_key, args.concurrency, output_path)


def cmd_generate(args: argparse.Namespace) -> None:
    """Handler for the 'generate' subcommand (matrix experiment mode)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] OPENROUTER_API_KEY not set.")
        sys.exit(1)

    # Load prompts from JSON (only the prompts array is needed)
    try:
        prompts, _, default_judge = load_test_file(args.json_file)
    except FileNotFoundError:
        console.print(f"[bold red]File not found:[/] {args.json_file}")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as exc:
        console.print(f"[bold red]Invalid test file:[/] {exc}")
        sys.exit(1)

    # Filter by categories if specified
    if args.categories:
        prompts = [p for p in prompts if p.category in args.categories]
        if not prompts:
            console.print("[bold red]No prompts match the specified categories.[/]")
            sys.exit(1)

    # Select tier defaults
    if args.tier == "paid":
        tier_sub_models = DEFAULT_NONFREE_SUB_MODELS
        tier_master_model = DEFAULT_NONFREE_MASTER_MODEL
    else:
        tier_sub_models = DEFAULT_SUB_MODELS
        tier_master_model = DEFAULT_MASTER_MODEL

    # Build model pool
    if args.model_pool:
        model_pool = [{"id": mid, "label": mid.split("/")[-1]} for mid in args.model_pool]
    else:
        model_pool = [
            {"id": m["id"], "label": m["label"]}
            for m in tier_sub_models
        ]

    # Resolve master model
    if args.master:
        master_model = {"id": args.master, "label": args.master.split("/")[-1]}
    else:
        master_model = dict(tier_master_model)

    # Parse phases
    phases = [int(p) for p in args.phases.split(",")] if args.phases else [1, 2, 3]

    # Resolve roles
    role_pool = list(ROLE_NAMES)

    console.print(f"\n[bold cyan]SynthesizAIr Matrix Generator[/]")
    tier_label = "[green]free[/]" if args.tier == "free" else "[yellow]paid[/]"
    console.print(f"  Tier         : {tier_label}")
    console.print(f"  Model pool   : {len(model_pool)} models")
    for m in model_pool:
        console.print(f"    {m['label']} ({m['id']})")
    console.print(f"  Master       : {master_model['label']}")
    console.print(f"  Phases       : {phases}")
    console.print(f"  Max per phase: {args.max_combinations}")

    # Generate combinations
    entries = generate_combinations(
        model_pool=model_pool,
        role_pool=role_pool,
        phases=phases,
        max_combinations=args.max_combinations,
        master_model=master_model,
    )

    if not entries:
        console.print("[bold red]No combinations generated. Check model pool size and phases.[/]")
        sys.exit(1)

    judge_model = _resolve_judge(args, default_judge)

    # Fetch pricing and estimate cost
    console.print("\n[dim]Fetching model pricing from OpenRouter...[/]")
    pricing = asyncio.run(fetch_model_pricing(api_key))
    cost_estimate = estimate_cost(entries, prompts, master_model, judge_model, pricing)

    # Write preview
    preview_path = args.preview or "test_matrix.json"
    estimated_calls = write_matrix_preview(
        entries, prompts, model_pool, master_model,
        args.categories, output_path=preview_path,
    )
    print_matrix_preview(entries, prompts, estimated_calls, cost_estimate=cost_estimate)
    console.print(f"\n  Preview written to: [green]{preview_path}[/]")

    # Confirmation gate
    if args.dry_run:
        console.print("\n[dim]--dry-run: stopping before execution.[/]")
        return

    cost_str = f", ~${cost_estimate['total']:.4f}" if cost_estimate["total"] > 0 else ""
    try:
        from rich.prompt import Prompt
        confirm = Prompt.ask(
            f"\nProceed with [bold]{len(prompts) * len(entries)}[/] runs "
            f"(~[bold]{estimated_calls}[/] API calls{cost_str})?",
            choices=["y", "n"],
            default="y",
        )
        if confirm != "y":
            console.print("[dim]Cancelled.[/]")
            return
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelled.[/]")
        return

    # Prepare combinations and metadata for the batch runner
    combinations = [e["combination"] for e in entries]
    combo_metadata = {
        e["combination"].name: {
            "combination_id": e["combination_id"],
            "phase": e["phase_label"],
        }
        for e in entries
    }
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = args.output or f"matrix_results_{ts}.csv"

    console.print(
        f"\n  Judge model  : "
        + (f"[green]{judge_model['label']}[/]" if judge_model else "[dim]disabled[/]")
    )
    console.print(f"  Concurrency  : {args.concurrency}")
    console.print(f"  Output       : {output_path}\n")

    _run_and_report(
        prompts, combinations, judge_model, api_key,
        args.concurrency, output_path,
        combo_metadata=combo_metadata,
        is_matrix=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation harness for SynthesizAIr.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run: original batch tester ──
    run_parser = subparsers.add_parser(
        "run",
        help="Run batch tests from a JSON file (original mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    run_parser.add_argument("json_file", help="Path to JSON test-prompts file")
    _add_common_args(run_parser)

    # ── generate: matrix experiment mode ──
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate & run a test matrix across model/role combinations",
    )
    gen_parser.add_argument("json_file", help="Path to JSON prompts file")
    gen_parser.add_argument(
        "--tier",
        choices=["free", "paid"],
        default="free",
        help="Model tier for defaults: 'free' or 'paid' (default: free)",
    )
    gen_parser.add_argument(
        "--model-pool",
        nargs="+",
        metavar="MODEL_ID",
        help="Model IDs for the pool (overrides --tier for sub-models)",
    )
    gen_parser.add_argument(
        "--master",
        metavar="MODEL_ID",
        help="Master model ID (overrides --tier for master)",
    )
    gen_parser.add_argument(
        "--phases",
        default="1,2,3",
        help="Comma-separated phase numbers to run (default: 1,2,3)",
    )
    gen_parser.add_argument(
        "--max-combinations",
        type=int,
        default=10,
        help="Max combinations per phase — budget cap (default: 10)",
    )
    gen_parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        help=f"Filter prompts to these categories (default: all). Choices: {CATEGORY_NAMES}",
    )
    gen_parser.add_argument(
        "--preview",
        metavar="PATH",
        help="Path for the preview JSON (default: test_matrix.json)",
    )
    gen_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and preview the matrix but do not execute runs",
    )
    _add_common_args(gen_parser)

    # Backward compatibility: if first arg is not a known subcommand,
    # treat the entire argv as the 'run' subcommand.
    known_commands = {"run", "generate", "-h", "--help"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
