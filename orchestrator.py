import asyncio
import dataclasses
import time

import httpx

from config import (
    OPENROUTER_CHAT_ENDPOINT,
    MAX_TOKENS_SUB,
    MAX_TOKENS_MASTER,
    MAX_TOKENS_DISAGREEMENT,
    TEMPERATURE_SUB,
    TEMPERATURE_MASTER,
    REQUEST_TIMEOUT_SECONDS,
    CATEGORIES,
    DISAGREEMENT_ABSENT_MARKER,
    DISAGREEMENT_PROMPT_TEMPLATE,
    ROLES,
    SYNTHESIS_PROMPT_TEMPLATE,
)


@dataclasses.dataclass
class ModelResult:
    model_id: str
    label: str
    content: str | None
    error: str | None
    elapsed_seconds: float
    role: str = ""


def _build_request_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/synthesizair",
        "X-Title": "SynthesizAIr",
    }


def _fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Move the system prompt into the first user message as a preamble."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]
    if not system_parts or not other:
        return other or messages
    preamble = "\n\n".join(system_parts)
    result = list(other)
    result[0] = {
        "role": result[0]["role"],
        "content": f"[INSTRUCTIONS]\n{preamble}\n[/INSTRUCTIONS]\n\n{result[0]['content']}",
    }
    return result


async def call_model(
    client: httpx.AsyncClient,
    model: dict,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    api_key: str,
) -> ModelResult:
    role = model.get("role", "")

    # Prepend system prompt when a recognised role is assigned
    if role and role in ROLES:
        full_messages = [{"role": "system", "content": ROLES[role]["system_prompt"]}] + messages
    else:
        full_messages = messages

    headers = _build_request_headers(api_key)
    start = time.monotonic()
    try:
        response = await client.post(
            OPENROUTER_CHAT_ENDPOINT,
            headers=headers,
            json={
                "model": model["id"],
                "messages": full_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return ModelResult(model["id"], model["label"], content, None, time.monotonic() - start, role)
    except httpx.HTTPStatusError as e:
        # Some models (e.g. Gemma) reject system/developer messages with a 400.
        # Retry by folding the system prompt into the user message instead.
        if e.response.status_code == 400 and any(
            hint in e.response.text.lower()
            for hint in ("developer instruction", "system message", "system role", "system prompt")
        ):
            fallback_messages = _fold_system_into_user(full_messages)
            try:
                response = await client.post(
                    OPENROUTER_CHAT_ENDPOINT,
                    headers=headers,
                    json={
                        "model": model["id"],
                        "messages": fallback_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ModelResult(model["id"], model["label"], content, None, time.monotonic() - start, role)
            except Exception as e2:
                return ModelResult(
                    model["id"], model["label"], None,
                    f"Fallback also failed: {e2}",
                    time.monotonic() - start, role,
                )
        return ModelResult(
            model["id"], model["label"], None,
            f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            time.monotonic() - start, role,
        )
    except Exception as e:
        return ModelResult(model["id"], model["label"], None, str(e), time.monotonic() - start, role)


def _build_analyses_block(sub_results: list, master_initial: ModelResult) -> tuple[str, int]:
    """Return (formatted analyses block, number of successful results)."""
    parts = []
    for r in sub_results:
        role_tag = f"{r.role.upper()} — " if r.role else ""
        header = f"[{role_tag}{r.label}]"
        if r.content:
            parts.append(f"{header}\n{r.content}")
        else:
            parts.append(f"{header} — FAILED: {r.error}")

    if master_initial.content:
        parts.append(f"[MASTER'S INDEPENDENT VIEW — {master_initial.label}]\n{master_initial.content}")
    else:
        parts.append(f"[MASTER'S INDEPENDENT VIEW — {master_initial.label}] — FAILED: {master_initial.error}")

    block = "\n\n---\n\n".join(parts)
    n_successful = sum(1 for r in sub_results + [master_initial] if r.content)
    return block, n_successful


async def _detect_disagreements(
    client: httpx.AsyncClient,
    question: str,
    sub_results: list,
    master_initial: ModelResult,
    master_model: dict,
    api_key: str,
) -> ModelResult:
    """Phase 1.5: ask the master to identify substantive disagreements between sub-models."""
    analyses_block, n_successful = _build_analyses_block(sub_results, master_initial)
    detect_prompt = DISAGREEMENT_PROMPT_TEMPLATE.format(
        n=n_successful,
        question=question,
        analyses=analyses_block,
        # placeholder names shown in the template instructions — not substituted literally
        role_a="Role A",
        role_b="Role B",
    )
    messages = [{"role": "user", "content": detect_prompt}]
    return await call_model(
        client, master_model, messages, MAX_TOKENS_DISAGREEMENT, TEMPERATURE_MASTER, api_key
    )


def _format_analyses(
    question: str,
    sub_results: list,
    master_initial: ModelResult,
    category: str | None = None,
    disagreements_text: str | None = None,
) -> str:
    analyses_block, n_successful = _build_analyses_block(sub_results, master_initial)

    if category and category in CATEGORIES:
        category_guidance = CATEGORIES[category]["synthesis_guidance"]
    else:
        category_guidance = (
            "Synthesize these perspectives into one comprehensive, definitive final answer. "
            "Weight each contribution by its cognitive role."
        )

    if disagreements_text:
        disagreement_section = (
            "\n\nIDENTIFIED TENSIONS — you must explicitly resolve or address each one:\n\n"
            + disagreements_text
        )
    else:
        disagreement_section = ""

    return SYNTHESIS_PROMPT_TEMPLATE.format(
        n=n_successful,
        question=question,
        analyses=analyses_block,
        category_guidance=category_guidance,
        disagreement_section=disagreement_section,
    )


async def run_synthesis(
    prompt: str,
    api_key: str,
    sub_models: list[dict],
    master_model: dict,
    on_result=None,
    category: str | None = None,
) -> dict:
    """
    Returns:
      {
        "sub_results": list[ModelResult],
        "master_initial": ModelResult,
        "disagreements": ModelResult,   # content is sentinel or bullet list
        "synthesis": ModelResult,
      }
    """
    user_message = [{"role": "user", "content": prompt}]

    async with httpx.AsyncClient() as client:
        # Phase 1: all 6 calls in parallel (each sub-model uses its role system prompt)
        coros = []
        for model in sub_models:
            coros.append(call_model(client, model, user_message, MAX_TOKENS_SUB, TEMPERATURE_SUB, api_key))
        # Master runs without a role system prompt — unbiased independent view
        coros.append(call_model(client, master_model, user_message, MAX_TOKENS_SUB, TEMPERATURE_SUB, api_key))

        pending = [asyncio.ensure_future(c) for c in coros]
        all_results: list[ModelResult] = []
        for fut in asyncio.as_completed(pending):
            result = await fut
            all_results.append(result)
            if on_result:
                on_result(result)

        sub_results = [r for r in all_results if r.model_id != master_model["id"]]
        master_initial = next(r for r in all_results if r.model_id == master_model["id"])

        # Phase 1.5: detect disagreements between sub-model responses
        disagreement_result = await _detect_disagreements(
            client, prompt, sub_results, master_initial, master_model, api_key
        )

        # Extract disagreement text only when real tensions were found
        disagreements_text: str | None = None
        if disagreement_result.content:
            stripped = disagreement_result.content.strip()
            if not stripped.upper().startswith(DISAGREEMENT_ABSENT_MARKER):
                disagreements_text = stripped

        # Phase 2: master synthesizes all responses, informed by detected tensions
        synthesis_prompt = _format_analyses(
            prompt, sub_results, master_initial,
            category=category,
            disagreements_text=disagreements_text,
        )
        synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
        synthesis_result = await call_model(
            client, master_model, synthesis_messages,
            MAX_TOKENS_MASTER, TEMPERATURE_MASTER, api_key,
        )

    return {
        "sub_results": sub_results,
        "master_initial": master_initial,
        "disagreements": disagreement_result,
        "synthesis": synthesis_result,
    }
