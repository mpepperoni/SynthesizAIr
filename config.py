OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_MODELS_ENDPOINT = f"{OPENROUTER_BASE_URL}/models"

# ---------------------------------------------------------------------------
# Cognitive roles
# ---------------------------------------------------------------------------

ROLES = {
    "Analytical": {
        "description": "Logical, evidence-based, systematic reasoning",
        "color": "blue",
        "system_prompt": (
            "You are an analytical thinker. Approach every question by breaking it into "
            "components, identifying the key variables, and reasoning step-by-step from "
            "evidence to conclusion. Be precise and structured. Distinguish clearly between "
            "what is known, what is inferred, and what is uncertain. Avoid unsupported claims."
        ),
    },
    "Devil's Advocate": {
        "description": "Challenges assumptions, surfaces counterarguments",
        "color": "red",
        "system_prompt": (
            "You are a devil's advocate. Your job is to challenge the most common or "
            "comfortable answer to any question. Identify hidden assumptions, expose "
            "weaknesses in popular positions, and argue for the strongest plausible "
            "counterpoint — even if you don't personally hold it. Surface edge cases, "
            "risks, and perspectives that are easy to overlook."
        ),
    },
    "Creative": {
        "description": "Lateral thinking, novel framings, unexpected angles",
        "color": "magenta",
        "system_prompt": (
            "You are a creative thinker. Approach every question with lateral thinking: "
            "draw analogies from unrelated fields, challenge implicit constraints, reframe "
            "the problem in unexpected ways, and surface non-obvious insights. Prioritise "
            "originality and intellectual surprise over convention. It is fine to be "
            "speculative — clearly mark speculation as such."
        ),
    },
    "Pragmatist": {
        "description": "Real-world focus, practical constraints, actionable advice",
        "color": "green",
        "system_prompt": (
            "You are a pragmatist. Cut through theory and focus on what works in the real "
            "world. Identify practical constraints, resource limits, implementation "
            "challenges, and likely failure modes. Favour actionable recommendations over "
            "abstract ideals. Ask: what would actually happen if this were tried? What "
            "tradeoffs must be accepted? What is the simplest path to a good outcome?"
        ),
    },
    "Synthesizer": {
        "description": "Connects ideas, finds meta-patterns, reconciles tensions",
        "color": "yellow",
        "system_prompt": (
            "You are a synthesizer. Look for the connections between ideas, identify "
            "meta-patterns, and reconcile apparent contradictions. Find the underlying "
            "principles that unite different perspectives on the question. Your goal is "
            "integrative understanding: show how disparate threads fit together into a "
            "coherent whole, and highlight what that larger picture reveals."
        ),
    },
}

ROLE_NAMES: list[str] = list(ROLES.keys())

# ---------------------------------------------------------------------------
# Prompt categories — each shapes how the master weights the roles
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Strategy/Planning": {
        "description": "Decisions, plans, roadmaps, competitive positioning",
        "color": "cyan",
        "synthesis_guidance": (
            "This is a STRATEGY/PLANNING question. Apply the following role weighting:\n"
            "- Pragmatist (HIGHEST): execution realities and practical constraints drive good strategy\n"
            "- Devil's Advocate (HIGH): stress-test the plan — where will it break down?\n"
            "- Analytical (NORMAL): use for evidence-grounded assessment of options\n"
            "- Synthesizer (NORMAL): connect strategic threads into a coherent direction\n"
            "- Creative (LOWER): novel angles are welcome but must serve practical ends\n"
            "Structure your answer around: recommended direction, key risks to anticipate, "
            "and concrete next steps."
        ),
    },
    "Research/Synthesis": {
        "description": "Information gathering, summarisation, knowledge synthesis",
        "color": "blue",
        "synthesis_guidance": (
            "This is a RESEARCH/SYNTHESIS question. Apply the following role weighting:\n"
            "- Analytical (HIGHEST): factual accuracy and logical rigour are primary\n"
            "- Synthesizer (HIGH): connect findings into a coherent, integrated understanding\n"
            "- Devil's Advocate (NORMAL): challenge sources and surface conflicting evidence\n"
            "- Creative (NORMAL): surface non-obvious cross-domain connections\n"
            "- Pragmatist (LOWER): practical implications matter but theory comes first\n"
            "Structure your answer around: what is well-established, where evidence is contested, "
            "and what an integrated reading of the evidence suggests."
        ),
    },
    "Analysis/Evaluation": {
        "description": "Comparing options, diagnosing problems, assessing trade-offs",
        "color": "yellow",
        "synthesis_guidance": (
            "This is an ANALYSIS/EVALUATION question. Apply the following role weighting:\n"
            "- Analytical (HIGHEST): systematic comparison and evidence-based reasoning are central\n"
            "- Devil's Advocate (HIGH): expose weaknesses in every option, not just the popular ones\n"
            "- Pragmatist (NORMAL): ground the evaluation in real-world feasibility\n"
            "- Synthesizer (NORMAL): identify criteria that reconcile apparent conflicts\n"
            "- Creative (LOWER): only invoke if it reveals a genuinely overlooked option\n"
            "Structure your answer around: clear evaluation criteria, how each option fares, "
            "and a reasoned conclusion."
        ),
    },
    "Creative/Brainstorming": {
        "description": "Ideation, exploration, imagination, open-ended possibility",
        "color": "magenta",
        "synthesis_guidance": (
            "This is a CREATIVE/BRAINSTORMING question. Apply the following role weighting:\n"
            "- Creative (HIGHEST): novel ideas, lateral leaps, and unexpected framings are the priority\n"
            "- Synthesizer (HIGH): weave ideas into themes and reveal surprising connections\n"
            "- Devil's Advocate (NORMAL): stress-test ideas to keep ideation honest and grounded\n"
            "- Analytical (LOWER): useful for structure, but don't let rigour kill imagination\n"
            "- Pragmatist (LOWER): note feasibility briefly but don't prune ideas prematurely\n"
            "Structure your answer around: the most promising ideas, unexpected angles, "
            "and how they could be developed further."
        ),
    },
}

CATEGORY_NAMES: list[str] = list(CATEGORIES.keys())

# ---------------------------------------------------------------------------
# Default model lineup — one role assigned per sub-model
# ---------------------------------------------------------------------------

DEFAULT_SUB_MODELS = [
    {"id": "mistralai/mistral-7b-instruct:free",          "label": "Mistral 7B",        "role": "Analytical"},
    {"id": "meta-llama/llama-3.2-3b-instruct:free",       "label": "Llama 3.2 3B",      "role": "Devil's Advocate"},
    {"id": "google/gemma-3-4b-it:free",                   "label": "Gemma 3 4B",         "role": "Creative"},
    {"id": "microsoft/phi-3-mini-128k-instruct:free",     "label": "Phi-3 Mini",         "role": "Pragmatist"},
    {"id": "deepseek/deepseek-r1-distill-qwen-1.5b:free", "label": "DeepSeek R1 1.5B",   "role": "Synthesizer"},
]

DEFAULT_MASTER_MODEL = {
    "id": "mistralai/mistral-small-3.1-24b-instruct:free",
    "label": "Mistral Small 24B (Master)",
}

# ---------------------------------------------------------------------------
# Request settings
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 60
MAX_TOKENS_SUB = 1024
MAX_TOKENS_MASTER = 2048
MAX_TOKENS_DISAGREEMENT = 600
TEMPERATURE_SUB = 0.7
TEMPERATURE_MASTER = 0.3

# Sentinel returned by the disagreement detector when no real divergence exists
DISAGREEMENT_ABSENT_MARKER = "NO_SIGNIFICANT_DISAGREEMENTS"

# ---------------------------------------------------------------------------
# Disagreement detection prompt — Phase 1.5, run before synthesis
# ---------------------------------------------------------------------------

DISAGREEMENT_PROMPT_TEMPLATE = """\
You are reviewing {n} independent responses to the same question, each produced by a \
model with a distinct cognitive role. Your job is to identify SUBSTANTIVE disagreements \
only — points where the responses genuinely conflict on facts, interpretations, \
recommendations, or conclusions. Ignore differences that are merely stylistic or a \
matter of emphasis.

QUESTION: {question}

RESPONSES:

{analyses}

If the responses are broadly consistent and any differences are minor, respond with \
exactly this single line:
NO_SIGNIFICANT_DISAGREEMENTS

Otherwise, identify 2–4 specific tension points using this exact format:

• [TENSION]: One sentence naming what is contested.
  [{role_a}] argues: their position in one sentence.
  [{role_b}] argues: their position in one sentence.

Be precise. Use the role names (Analytical, Devil's Advocate, Creative, Pragmatist, \
Synthesizer) to identify who holds each position. Keep every bullet to 3 sentences max.\
"""

# ---------------------------------------------------------------------------
# Synthesis prompt — references roles so master can weight perspectives
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_TEMPLATE = """\
You have received {n} independent analyses of the following question. Each analysis was \
produced by a model operating under a specific cognitive role that shaped its perspective.

QUESTION: {question}

ANALYSES:

{analyses}

INSTRUCTIONS FOR SYNTHESIS:
You are the master orchestrator. {category_guidance}

As a reminder, the cognitive roles provide:
- Analytical: factual grounding and logical structure
- Devil's Advocate: blind spots and untested assumptions
- Creative: novel framings and non-obvious insights
- Pragmatist: real-world applicability and practical constraints
- Synthesizer: integrative connections and meta-patterns
- Your own independent view: baseline to validate and anchor the above

Where perspectives align, state the consensus confidently. Where they diverge, reason \
through which position is best supported given the question's context. Do not merely \
summarise — produce a definitive, well-structured answer.
{disagreement_section}\
"""
