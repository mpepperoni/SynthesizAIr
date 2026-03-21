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
import json
import os
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
    DEFAULT_MASTER_MODEL,
    DEFAULT_SUB_MODELS,
    DISAGREEMENT_ABSENT_MARKER,
    OPENROUTER_CHAT_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
)
from orchestrator import run_synthesis

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
) -> list[RunResult]:
    semaphore = asyncio.Semaphore(concurrency)
    ts_tag = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    total = len(prompts) * len(combinations)
    task_id = progress.add_task("Starting…", total=total)

    async def _run_one(p: TestPrompt, combo: Combination) -> RunResult:
        async with semaphore:
            run_id = f"{p.id}__{combo.name}__{ts_tag}"
            progress.update(
                task_id,
                description=f"[cyan]{p.id}[/] × [yellow]{combo.name}[/]",
            )
            result = await run_single(p, combo, judge_model, api_key, run_id)
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
    "run_id", "timestamp", "prompt_id", "prompt_snippet", "category", "combination",
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
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation harness for SynthesizAIr.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("json_file", help="Path to JSON test-prompts file")
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
    args = parser.parse_args()

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

    if args.judge:
        judge_model = {"id": args.judge, "label": args.judge}
    if args.no_judge:
        judge_model = None

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
                api_key, args.concurrency, progress,
            )
        )

    print_terminal_summary(results)
    write_csv(results, output_path)


if __name__ == "__main__":
    main()
