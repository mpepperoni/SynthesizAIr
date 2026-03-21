# SynthesizAIr

A multi-model AI orchestration system that leverages five cognitively-distinct sub-models to synthesize sophisticated, multi-perspective answers to complex questions. Instead of relying on a single model, SynthesizAIr distributes a question across specialized models, identifies disagreements, and reconciles perspectives into a unified final answer.

## How It Works

SynthesizAIr runs a three-phase pipeline:

1. **Parallel Sub-Model Queries** — Five sub-models, each assigned a distinct cognitive role, process the question simultaneously alongside a master model's independent view.
2. **Disagreement Detection** — The master model reviews all responses and identifies substantive tensions between perspectives.
3. **Final Synthesis** — The master model reconciles all perspectives, guided by category-specific weighting, into one definitive answer.

### Cognitive Roles

| Role | Focus |
|------|-------|
| **Analytical** | Logical, evidence-based, systematic reasoning |
| **Devil's Advocate** | Challenges assumptions, surfaces counterarguments |
| **Creative** | Lateral thinking, novel framings, unexpected angles |
| **Pragmatist** | Real-world focus, practical constraints, actionable advice |
| **Synthesizer** | Connects ideas, finds meta-patterns, reconciles tensions |

### Prompt Categories

Questions can be categorized to adjust how the master model weights each role:

- **Strategy/Planning** — Weighs Pragmatist highest
- **Research/Synthesis** — Weighs Analytical highest
- **Analysis/Evaluation** — Weighs Analytical and Devil's Advocate highest
- **Creative/Brainstorming** — Weighs Creative highest

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

### Configuration

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-<your-key-here>
```

Or set it as an environment variable:

```bash
export OPENROUTER_API_KEY=sk-or-v1-<your-key-here>
```

## Usage

### Interactive CLI

```bash
python synthesizer.py
```

Starts a terminal UI where you can select categories, ask questions, and view live responses from all sub-models plus the final synthesis.

### REST API

```bash
uvicorn api:app --reload
```

Exposes three endpoints at `http://localhost:8000`:

**POST /synthesize** — Run the full orchestration pipeline.

```json
{
  "prompt": "Should we focus on growth or profitability?",
  "category": "Strategy/Planning"
}
```

**GET /models** — Returns available free models from OpenRouter and current defaults.

**GET /roles** — Returns role and category definitions.

Authentication is via the `X-OpenRouter-Key` header or the `OPENROUTER_API_KEY` environment variable.

### Batch Testing

```bash
python batch_tester.py test_prompts.json
python batch_tester.py test_prompts.json -o results.csv -c 2
python batch_tester.py test_prompts.json --no-judge
```

Runs synthesis across multiple prompts and model combinations, optionally scoring results with a judge model. Outputs a CSV with detailed metrics.

Options:
- `-o/--output` — Output CSV file path
- `-c/--concurrency` — Max parallel runs (default: 1)
- `--judge` — Override judge model
- `--no-judge` — Skip judge scoring

## Default Models

All defaults use free-tier OpenRouter models:

| Role | Model |
|------|-------|
| Analytical | Mistral 7B |
| Devil's Advocate | Llama 3.2 3B |
| Creative | Gemma 3 4B |
| Pragmatist | Phi-3 Mini |
| Synthesizer | DeepSeek R1 1.5B |
| **Master** | **Mistral Small 24B** |

Models can be customized at runtime through the CLI, API request body, or batch test JSON.

## Configuration Reference

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `REQUEST_TIMEOUT_SECONDS` | 60 | Timeout per API call |
| `MAX_TOKENS_SUB` | 1024 | Max tokens for sub-model responses |
| `MAX_TOKENS_MASTER` | 2048 | Max tokens for master synthesis |
| `MAX_TOKENS_DISAGREEMENT` | 600 | Max tokens for disagreement detection |
| `TEMPERATURE_SUB` | 0.7 | Creativity for sub-models |
| `TEMPERATURE_MASTER` | 0.3 | Precision for master model |

## Project Structure

```
├── api.py              # FastAPI REST interface
├── orchestrator.py     # Core multi-model orchestration pipeline
├── synthesizer.py      # Interactive CLI with rich terminal UI
├── batch_tester.py     # Batch evaluation harness with judge scoring
├── config.py           # Models, roles, categories, prompts, hyperparameters
├── test_prompts.json   # Sample test prompts
├── requirements.txt    # Python dependencies
└── LICENSE             # GPL-3.0
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
