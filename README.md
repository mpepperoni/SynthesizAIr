# SynthesizAIr

> *Stop opening five browser tabs. Let them argue it out for you.*

A multi-model AI orchestration system that routes a single question through five cognitively distinct sub-models in parallel, detects where they disagree, then has a master model synthesize all perspectives into one authoritative answer.

Built by [@mpepperoni](https://github.com/mpepperoni) — contributions welcome.

---

## Why This Exists

Anyone who uses LLMs seriously knows the workflow: open ChatGPT, ask a question, open Claude, ask the same question, maybe check one more. Then mentally synthesize the differences yourself.

That last step — the synthesis — is where the real value is. And it's the part you're doing manually.

SynthesizAIr automates it. Five models, five cognitive lenses, one synthesized answer. The disagreements between models aren't noise — they're signal. The tool surfaces them explicitly rather than papering over them.

This is an early-stage project. The goal isn't to compete with Poe or OpenRouter's playground — it's to give the community a foundation to build on. Fork it, extend it, make it better.

---

## How It Works

SynthesizAIr runs a three-phase pipeline:

```
Phase 1  →  Five sub-models process your question simultaneously
            Each model is assigned a distinct cognitive role
            All calls are async — no waiting for one to finish before the next starts

Phase 1.5 →  Disagreement detection
             Master model reviews all five outputs
             Identifies substantive tensions between perspectives
             (Most ensemble tools skip this step — this one doesn't)

Phase 2  →  Master synthesis
            Reconciles all perspectives into one final answer
            Weighted by prompt category
            Disagreements surfaced explicitly, not buried
```

### The Role System

The role system is what separates this from just running the same prompt five times. Each sub-model is given a cognitive persona that genuinely changes how it approaches the question:

| Role | What It Does |
|---|---|
| **Analytical** | Systematic, evidence-based breakdown. Components, logic, structure. |
| **Devil's Advocate** | Actively challenges assumptions. Asks what could go wrong, what's being glossed over. |
| **Creative** | Lateral thinking, unexpected angles, "what if" framings. |
| **Pragmatist** | Cuts theory. Focuses on what actually works in the real world. |
| **Synthesizer** | Finds the common thread. Reduces complexity. Distills to essentials. |

The Devil's Advocate role in particular tends to surface the uncomfortable truths that a single LLM query would soften or skip entirely.

### Prompt Categories

Selecting a category adjusts how the master model weights each role during synthesis:

| Category | Highest Weight |
|---|---|
| Strategy/Planning | Pragmatist + Devil's Advocate |
| Research/Synthesis | Analytical |
| Analysis/Evaluation | Analytical + Devil's Advocate |
| Creative/Brainstorming | Creative |

---

## Early Findings

Three test runs across both free-tier and paid OpenRouter models. These are real outputs, not cherry-picked demos.

### Model Panels Tested

**Free Panel** — all `:free` tier OpenRouter models
| Role | Model |
|---|---|
| Analytical | Nemotron Nano 30B |
| Devil's Advocate | Step 3.5 Flash |
| Creative | MiniMax M2.5 |
| Pragmatist | LFM 1.2B Instruct |
| Synthesizer | Trinity Large |
| Master | Nemotron Super 120B |

**Paid Panel** — frontier models via OpenRouter
| Role | Model |
|---|---|
| Devil's Advocate | Gemini 2.5 Pro |
| Creative | GPT-4.1 |
| Pragmatist | Llama 4 Maverick |
| Synthesizer | DeepSeek R1 |
| Master | Claude Sonnet 4 |

---

### Run 1 — Authentic Chinese Noodle Recipe (Creative/Brainstorming) — Free Panel

Five models diverged immediately on dish choice — Analytical went Lanzhou beef noodle soup, Creative went Dan Dan noodles, Pragmatist focused on accessible substitutions. The master synthesized a hybrid: Lanzhou broth architecture with a Dan Dan sesame-chili sauce layer. It attributed each element to the sub-model that contributed it. The result was a coherent, cookable recipe that no single model produced independently.

**Key observation:** Role diversity mattered more than model diversity here. The Creative role's willingness to pick a completely different dish gave the master real material to work with.

---

### Run 2 — Career Transition: Technical Trade → Network Engineering (Strategy/Planning) — Free Panel

The Devil's Advocate role produced the most valuable output in this run. While other models gave structured, encouraging advice about certifications and skill gaps, the Devil's Advocate pushed back hard on the certification-first mindset — arguing that a CCNA without hands-on depth is just paper, and that real transition means building systems thinking, not collecting credentials.

The master synthesis incorporated this tension explicitly, pairing each recommendation with a risk column drawn from the Devil's Advocate output.

**Key observation:** The Devil's Advocate role's effectiveness scales with prompt stakes. Career decisions have real consequences. The role produced more valuable output here than it did on the recipe prompt, where "authenticity" is philosophical rather than consequential.

**Early hypothesis for batch testing:** Devil's Advocate role impact likely correlates with category:
```
High impact   →  Strategy/Planning, Analysis/Evaluation
Medium impact →  Research/Synthesis
Lower impact  →  Creative/Brainstorming
```

---

### Run 3 — Teaching Math to 5-6 Year Olds (Creative/Brainstorming) — Paid Panel

The quality jump from free to paid panel was immediately noticeable. Three genuine pedagogical tensions surfaced in disagreement detection — structure vs. exploration, concrete vs. abstract progression, and disciplinary vs. interdisciplinary framing. These aren't model artifacts; they're real debates in early childhood education that the system identified unprompted.

The master synthesis produced concepts that no single sub-model generated — a "Math Detective" framework resolving the structure/exploration tension, and "Mathematical Empathy" (numbers have friendships, shapes have families) as a novel framing for abstract concepts. The closing meta-insight — *"children don't learn math, they discover it was there all along"* — emerged purely from synthesis, not from any individual output.

**Cost:** $0.0714 for the full paid panel run — six model calls including master synthesis and disagreement detection.

**Key observation:** The paid panel produced richer disagreement and more original synthesis. The free panel is solid for development and testing architecture. The paid panel is where the output quality justifies the use case.

**Cross-run finding:** The best synthesis outputs occur when sub-models disagree on fundamentals, not just details. Run 1 disagreed on dish choice. Run 2 disagreed on what certifications actually mean. Run 3 disagreed on the philosophy of how children learn. Substantive disagreement between roles is the feature, not a bug.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, an [OpenRouter](https://openrouter.ai/) API key (free tier works).

```bash
# .env file or environment variable
OPENROUTER_API_KEY=sk-or-v1-<your-key-here>
```

---

## Usage

### Interactive CLI

```bash
python synthesizer.py
```

Starts a terminal UI. Select a category, enter your question, watch all six models respond in real time, then get the synthesized answer.

Two modes available at startup:
- **Auto** — Uses default model/role assignments
- **Custom** — Fetches full free model list from OpenRouter, assign any model to any role

### REST API

```bash
uvicorn api:app --reload
```

Exposes three endpoints at `http://localhost:8000`:

**POST /synthesize**
```json
{
  "prompt": "Should we focus on growth or profitability?",
  "category": "Strategy/Planning"
}
```

**GET /models** — Available models and current defaults

**GET /roles** — Role and category definitions

Auth via `X-OpenRouter-Key` header or `OPENROUTER_API_KEY` env var. Full interactive docs at `/docs`.

### Batch Testing

```bash
python batch_tester.py run test_prompts.json
python batch_tester.py run test_prompts.json -o results.csv -c 2
python batch_tester.py run test_prompts.json --no-judge
```

Runs synthesis across multiple prompts, optionally scoring with a judge model. Outputs CSV with detailed metrics per run.

### Matrix Experiments

```bash
# Preview before spending credits
python batch_tester.py generate test_prompts.json --dry-run

# Run full matrix
python batch_tester.py generate test_prompts.json --max-combinations 5 --phases 1,2,3
```

Runs three experiment phases:

| Phase | Name | Tests |
|---|---|---|
| 1 | Role Isolation | Same model × 5, all different roles — does the role system actually work? |
| 2 | Model Diversity | Different models, same role — does model choice matter independently? |
| 3 | Full Combination | Different models + different roles — what's the winning config? |

Generates a `test_matrix.json` preview showing every combination and estimated API call count before executing. Budget cap via `--max-combinations` prevents runaway costs.

After all runs, summary reports best combination per category, per phase findings, and overall winner.

---

## Default Models

Two built-in panels selectable at startup. All models swappable at runtime via CLI, API, or batch config.

**Free Panel** (default) — no API credits required
| Role | Model |
|---|---|
| Analytical | Nemotron Nano 30B |
| Devil's Advocate | Step 3.5 Flash |
| Creative | MiniMax M2.5 |
| Pragmatist | LFM 1.2B Instruct |
| Synthesizer | Trinity Large |
| **Master** | **Nemotron Super 120B** |

**Paid Panel** — frontier models, higher output quality (~$0.07/run)
| Role | Model |
|---|---|
| Devil's Advocate | Gemini 2.5 Pro |
| Creative | GPT-4.1 |
| Pragmatist | Llama 4 Maverick |
| Synthesizer | DeepSeek R1 |
| **Master** | **Claude Sonnet 4** |

---

## Project Structure

```
├── orchestrator.py     # Core async pipeline: Phase 1 → 1.5 → 2
├── synthesizer.py      # Interactive CLI with Rich terminal UI
├── batch_tester.py     # Batch evaluation + combination generator + judge scoring
├── api.py              # FastAPI wrapper (no business logic duplication)
├── config.py           # Models, roles, categories, prompts, hyperparameters
├── test_prompts.json   # 8 sample prompts across 4 categories
├── requirements.txt
└── LICENSE             # GPL-3.0
```

The architecture is intentionally layered — `orchestrator.py` contains all business logic, `synthesizer.py` and `api.py` are thin wrappers. Adding a web UI means touching only `api.py`.

---

## Configuration Reference

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `REQUEST_TIMEOUT_SECONDS` | 60 | Timeout per API call |
| `MAX_TOKENS_SUB` | 1024 | Max tokens for sub-model responses |
| `MAX_TOKENS_MASTER` | 8192 | Max tokens for master synthesis |
| `MAX_TOKENS_DISAGREEMENT` | 600 | Max tokens for disagreement detection |
| `TEMPERATURE_SUB` | 0.7 | Creativity for sub-models |
| `TEMPERATURE_MASTER` | 0.3 | Precision for master model |

---

## Roadmap

- [ ] Web UI (React frontend over existing FastAPI)
- [ ] Combination generator results dashboard
- [ ] Per-category optimal config recommendations from batch testing
- [ ] Role/category fit scoring dimension in batch tester
- [ ] Persistent run history and comparison

---

## Where This Works Best

SynthesizAIr's value scales with question complexity and ambiguity. The more a question benefits from multiple perspectives, the more the ensemble approach pays off.

**High value:** Business strategy, career decisions, research synthesis, complex tradeoff analysis, anything where "it depends" is the honest answer.

**Lower value:** Simple factual lookups, math, code execution, highly personal/contextual questions.

If your question has a single correct answer, this is overkill. If your question has five defensible answers and you need to think through all of them — this is the tool.

---

## Contributing

Early stage, all contributions welcome. Some natural starting points:

- Additional cognitive roles
- Web UI implementation
- Improved master synthesis prompt
- Role/category fit scoring
- Alternative LLM provider support beyond OpenRouter

Open an issue or submit a PR.

---

## License

[GNU General Public License v3.0](LICENSE)
