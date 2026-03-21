"""
api.py — FastAPI HTTP interface for SynthesizAIr.

All business logic lives in orchestrator.py and config.py; this module only
handles HTTP concerns: request validation, serialisation, and error mapping.

Start the server:
    uvicorn api:app --reload                     # development
    uvicorn api:app --host 0.0.0.0 --port 8000   # production

Authentication (in priority order per request):
  1. X-OpenRouter-Key request header  (useful for per-user keys from the web UI)
  2. OPENROUTER_API_KEY environment variable

Endpoints:
  POST /synthesize  — run the full multi-model orchestration pipeline
  GET  /models      — list available OpenRouter free models + current defaults
  GET  /roles       — return role definitions and prompt categories
"""

import os
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import (
    CATEGORIES,
    CATEGORY_NAMES,
    DEFAULT_MASTER_MODEL,
    DEFAULT_SUB_MODELS,
    DISAGREEMENT_ABSENT_MARKER,
    ROLE_NAMES,
    ROLES,
)
from orchestrator import ModelResult, run_synthesis
from synthesizer import fetch_free_models

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SynthesizAIr",
    description=(
        "Multi-model AI orchestrator: 5 cognitively-distinct sub-models feed "
        "a master synthesizer via OpenRouter."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Dependency — resolve the OpenRouter API key for every request that needs it
# ---------------------------------------------------------------------------


async def resolve_api_key(
    x_openrouter_key: str | None = Header(default=None),
) -> str:
    key = x_openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "No OpenRouter API key available. "
                "Set the OPENROUTER_API_KEY environment variable "
                "or pass the X-OpenRouter-Key request header."
            ),
        )
    return key


# ---------------------------------------------------------------------------
# Shared sub-schemas
# ---------------------------------------------------------------------------


class SubModelIn(BaseModel):
    id: str = Field(..., examples=["mistralai/mistral-7b-instruct:free"])
    label: str = Field(..., examples=["Mistral 7B"])
    role: str = Field(..., examples=["Analytical"])

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ROLE_NAMES:
            raise ValueError(f"role must be one of: {ROLE_NAMES}")
        return v


class MasterModelIn(BaseModel):
    id: str = Field(..., examples=["mistralai/mistral-small-3.1-24b-instruct:free"])
    label: str = Field(..., examples=["Mistral Small 24B"])


# ---------------------------------------------------------------------------
# POST /synthesize
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["Should we focus on growth or profitability?"])
    category: str | None = Field(
        default=None,
        examples=["Strategy/Planning"],
        description=f"One of: {CATEGORY_NAMES}. Shapes how the master weights the five roles.",
    )
    sub_models: list[SubModelIn] | None = Field(
        default=None,
        description="Five sub-models with roles assigned. Defaults to the built-in lineup.",
    )
    master_model: MasterModelIn | None = Field(
        default=None,
        description="Master synthesis model. Defaults to the built-in master.",
    )

    @field_validator("category")
    @classmethod
    def category_must_be_valid(cls, v: str | None) -> str | None:
        if v is not None and v not in CATEGORY_NAMES:
            raise ValueError(f"category must be one of: {CATEGORY_NAMES}")
        return v

    @field_validator("sub_models")
    @classmethod
    def sub_models_must_have_five(cls, v: list | None) -> list | None:
        if v is not None and len(v) != 5:
            raise ValueError("sub_models must contain exactly 5 entries (one per role)")
        return v


class ModelResultOut(BaseModel):
    model_id: str
    label: str
    role: str
    content: str | None
    error: str | None
    elapsed_seconds: float


class DisagreementsOut(BaseModel):
    found: bool = Field(description="True when the master detected substantive tensions between sub-models")
    content: str | None = Field(description="Bullet-list of tensions, or null when none found")


class SynthesizeResponse(BaseModel):
    synthesis: str | None = Field(description="The final synthesized answer")
    synthesis_error: str | None = Field(description="Set when the synthesis call itself failed")
    sub_results: list[ModelResultOut]
    master_initial: ModelResultOut
    disagreements: DisagreementsOut
    elapsed_seconds: float = Field(description="Total wall-clock time for the full pipeline")


def _model_result_to_out(r: ModelResult) -> ModelResultOut:
    return ModelResultOut(
        model_id=r.model_id,
        label=r.label,
        role=r.role,
        content=r.content,
        error=r.error,
        elapsed_seconds=round(r.elapsed_seconds, 2),
    )


@app.post(
    "/synthesize",
    response_model=SynthesizeResponse,
    summary="Run the full multi-model orchestration pipeline",
    response_description="Synthesis result plus individual sub-model responses and detected disagreements",
)
async def synthesize(
    body: SynthesizeRequest,
    api_key: str = Depends(resolve_api_key),
) -> SynthesizeResponse:
    sub_models = (
        [m.model_dump() for m in body.sub_models]
        if body.sub_models
        else list(DEFAULT_SUB_MODELS)
    )
    master_model = (
        body.master_model.model_dump()
        if body.master_model
        else dict(DEFAULT_MASTER_MODEL)
    )

    t0 = time.monotonic()
    try:
        outcome = await run_synthesis(
            body.prompt,
            api_key,
            sub_models,
            master_model,
            on_result=None,
            category=body.category,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestration error: {exc}") from exc

    elapsed = round(time.monotonic() - t0, 2)

    dr = outcome["disagreements"]
    disagree_found = (
        dr is not None
        and dr.content is not None
        and not dr.content.strip().upper().startswith(DISAGREEMENT_ABSENT_MARKER)
    )

    sr = outcome["synthesis"]

    return SynthesizeResponse(
        synthesis=sr.content,
        synthesis_error=sr.error if not sr.content else None,
        sub_results=[_model_result_to_out(r) for r in outcome["sub_results"]],
        master_initial=_model_result_to_out(outcome["master_initial"]),
        disagreements=DisagreementsOut(
            found=disagree_found,
            content=dr.content.strip() if disagree_found else None,
        ),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    id: str
    label: str


class DefaultModels(BaseModel):
    sub_models: list[dict]
    master_model: dict


class ModelsResponse(BaseModel):
    defaults: DefaultModels
    available: list[ModelEntry] = Field(
        description="All :free models from OpenRouter, sorted by name. Empty if the fetch failed."
    )
    available_fetch_error: bool = Field(
        default=False,
        description="True when the OpenRouter model list could not be retrieved",
    )


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available OpenRouter free models and the current default lineup",
)
async def list_models(api_key: str = Depends(resolve_api_key)) -> ModelsResponse:
    available = await fetch_free_models(api_key)
    return ModelsResponse(
        defaults=DefaultModels(
            sub_models=list(DEFAULT_SUB_MODELS),
            master_model=dict(DEFAULT_MASTER_MODEL),
        ),
        available=[ModelEntry(id=m["id"], label=m["label"]) for m in available],
        available_fetch_error=len(available) == 0,
    )


# ---------------------------------------------------------------------------
# GET /roles
# ---------------------------------------------------------------------------


class RoleDefinition(BaseModel):
    description: str
    color: str


class CategoryDefinition(BaseModel):
    description: str
    color: str


class RolesResponse(BaseModel):
    roles: dict[str, RoleDefinition]
    role_names: list[str] = Field(description="Ordered list of role names (same order as the default sub-model lineup)")
    categories: dict[str, CategoryDefinition]
    category_names: list[str] = Field(description="Ordered list of category names")


@app.get(
    "/roles",
    response_model=RolesResponse,
    summary="Return cognitive role definitions and prompt categories",
)
async def list_roles() -> RolesResponse:
    return RolesResponse(
        roles={
            name: RoleDefinition(
                description=meta["description"],
                color=meta["color"],
            )
            for name, meta in ROLES.items()
        },
        role_names=ROLE_NAMES,
        categories={
            name: CategoryDefinition(
                description=meta["description"],
                color=meta["color"],
            )
            for name, meta in CATEGORIES.items()
        },
        category_names=CATEGORY_NAMES,
    )
