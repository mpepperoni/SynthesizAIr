import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from config import (
    CATEGORIES,
    CATEGORY_NAMES,
    DEFAULT_MASTER_MODEL,
    DEFAULT_NONFREE_MASTER_MODEL,
    DEFAULT_NONFREE_SUB_MODELS,
    DEFAULT_SUB_MODELS,
    DISAGREEMENT_ABSENT_MARKER,
    OPENROUTER_MODELS_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    ROLES,
    ROLE_NAMES,
)
from orchestrator import ModelResult, run_synthesis

load_dotenv()

console = Console()


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        console.print(
            "[bold red]Error:[/] OPENROUTER_API_KEY not set.\n"
            "Set it in your environment or create a .env file with:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-..."
        )
        sys.exit(1)
    return key


def role_markup(role: str) -> str:
    """Return a Rich markup string for a role name, coloured by its config colour."""
    if role and role in ROLES:
        color = ROLES[role]["color"]
        return f"[{color}]{role}[/]"
    return role


async def fetch_free_models(api_key: str) -> list[dict]:
    """Fetch all :free models from OpenRouter and return sorted list."""
    all_models = await fetch_models(api_key)
    return [m for m in all_models if m["id"].endswith(":free")]


async def fetch_models(api_key: str) -> list[dict]:
    """Fetch all models from OpenRouter and return sorted list."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                OPENROUTER_MODELS_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            models = [
                {"id": m["id"], "label": m.get("name", m["id"])}
                for m in data.get("data", [])
            ]
            models.sort(key=lambda m: m["label"].lower())
            return models
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch model list: {e}[/]")
            return []


def display_model_list(models: list[dict]) -> None:
    table = Table(show_header=True, header_style="bold cyan", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Label")
    table.add_column("Model ID", style="dim")
    table.add_column("Tier", width=6)
    for i, m in enumerate(models, 1):
        tier = "[green]free[/]" if m["id"].endswith(":free") else "[dim]paid[/]"
        table.add_row(str(i), m["label"], m["id"], tier)
    console.print(table)


def display_roles() -> None:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Role", min_width=18)
    table.add_column("Focus", style="dim")
    for i, name in enumerate(ROLE_NAMES, 1):
        color = ROLES[name]["color"]
        table.add_row(str(i), f"[{color}]{name}[/]", ROLES[name]["description"])
    console.print(table)


def select_category() -> str | None:
    """Prompt the user to pick a prompt category. Returns the category name, or None to quit."""
    console.print("\n[bold]Prompt category:[/]")
    for i, name in enumerate(CATEGORY_NAMES, 1):
        color = CATEGORIES[name]["color"]
        desc = CATEGORIES[name]["description"]
        console.print(f"  [cyan][{i}][/] [{color}]{name}[/]  [dim]{desc}[/]")
    console.print(f"  [dim][q][/] [dim]Quit[/]")
    valid = [str(i) for i in range(1, len(CATEGORY_NAMES) + 1)] + ["q"]
    choice = Prompt.ask("\nCategory", choices=valid)
    if choice == "q":
        return None
    return CATEGORY_NAMES[int(choice) - 1]


def parse_indices(raw: str, max_idx: int) -> list[int] | None:
    """Parse comma-separated 1-based indices. Returns None on invalid input."""
    try:
        indices = [int(x.strip()) for x in raw.split(",")]
        if any(i < 1 or i > max_idx for i in indices):
            return None
        return indices
    except ValueError:
        return None


async def _pick_roles_for_models(sub_models: list[dict]) -> list[dict]:
    """Interactively assign a unique cognitive role to each sub-model. Returns sub_models with roles set."""
    console.print("\n[bold]Assign a cognitive role to each sub-model:[/]\n")
    display_roles()
    console.print()

    assigned_roles: list[str] = []
    remaining_roles = list(ROLE_NAMES)

    for i, m in enumerate(sub_models):
        available = [(j + 1, r) for j, r in enumerate(ROLE_NAMES) if r in remaining_roles]
        available_str = "  ".join(
            f"[{ROLES[r]['color']}][{idx}] {r}[/]" for idx, r in available
        )
        console.print(f"[bold]Sub-model {i + 1}:[/] {m['label']}")
        console.print(f"Available: {available_str}")

        valid_nums = [str(idx) for idx, _ in available]
        while True:
            raw = Prompt.ask(f"  Role for {m['label']}", choices=valid_nums)
            selected_role = ROLE_NAMES[int(raw) - 1]
            if selected_role not in remaining_roles:
                console.print("[red]That role is already taken. Pick another.[/]")
                continue
            assigned_roles.append(selected_role)
            remaining_roles.remove(selected_role)
            break

        console.print()

    for m, role in zip(sub_models, assigned_roles):
        m["role"] = role

    return sub_models


async def _auto_configure(
    sub_models: list[dict], master_model: dict
) -> tuple[list[dict], dict]:
    """Auto-mode sub-menu: auto roles (default) or custom role assignment."""
    console.print("\n[bold]Role assignment:[/]")
    console.print("  [cyan][1][/] Auto — use pre-assigned roles [dim](recommended)[/]")
    console.print("  [cyan][2][/] Custom — choose which role each model plays\n")

    role_choice = Prompt.ask("Choice", choices=["1", "2"], default="1")

    if role_choice == "1":
        return sub_models, master_model

    # Custom role assignment on the auto model set
    sub_copy = [dict(m) for m in sub_models]
    sub_copy = await _pick_roles_for_models(sub_copy)
    return sub_copy, master_model


async def configure_models(api_key: str) -> tuple[list[dict], dict]:
    """Interactive model + role configuration. Returns (sub_models, master_model)."""
    console.print("\n[bold]Model configuration:[/]")
    console.print("  [cyan][1][/] Auto (free)     — free models with pre-assigned roles [dim](recommended)[/]")
    console.print("  [cyan][2][/] Auto (non-free)  — higher-quality paid models")
    console.print("  [cyan][3][/] Custom           — choose models and assign roles manually\n")

    choice = Prompt.ask("Choice", choices=["1", "2", "3"], default="1")

    if choice == "1":
        return await _auto_configure(
            [dict(m) for m in DEFAULT_SUB_MODELS], dict(DEFAULT_MASTER_MODEL)
        )

    if choice == "2":
        return await _auto_configure(
            [dict(m) for m in DEFAULT_NONFREE_SUB_MODELS], dict(DEFAULT_NONFREE_MASTER_MODEL)
        )

    # Custom mode — fetch all available models
    console.print("\n[dim]Fetching available models from OpenRouter...[/]")
    all_models = await fetch_models(api_key)

    if not all_models:
        console.print("[yellow]Could not load model list. Falling back to defaults.[/]")
        return DEFAULT_SUB_MODELS, DEFAULT_MASTER_MODEL

    console.print(f"\n[bold]Available models[/] ({len(all_models)} total):\n")
    display_model_list(all_models)

    # Pick 5 sub-models
    while True:
        raw = Prompt.ask(
            "\nPick [bold]5 sub-models[/] (comma-separated numbers, e.g. [dim]1,3,7,12,15[/])"
        )
        indices = parse_indices(raw, len(all_models))
        if indices is None or len(indices) != 5:
            console.print("[red]Please enter exactly 5 valid numbers.[/]")
            continue
        sub_models = [dict(all_models[i - 1]) for i in indices]
        break

    # Pick master model
    while True:
        raw = Prompt.ask("\nPick [bold]1 master model[/] (single number)")
        indices = parse_indices(raw, len(all_models))
        if indices is None or len(indices) != 1:
            console.print("[red]Please enter exactly 1 valid number.[/]")
            continue
        master_model = all_models[indices[0] - 1]
        break

    # Role assignment sub-menu
    console.print("\n[bold]Role assignment:[/]")
    console.print("  [cyan][1][/] Auto — assign roles automatically")
    console.print("  [cyan][2][/] Custom — choose which role each model plays\n")

    role_choice = Prompt.ask("Choice", choices=["1", "2"], default="1")

    if role_choice == "1":
        for m, role in zip(sub_models, ROLE_NAMES):
            m["role"] = role
    else:
        sub_models = await _pick_roles_for_models(sub_models)

    # Confirm
    console.print("[bold]Selected configuration:[/]")
    for m in sub_models:
        console.print(f"  {role_markup(m['role']):<30} {m['label']}")
    console.print(f"  [yellow]Master[/]                         {master_model['label']}")

    confirm = Prompt.ask("\nProceed?", choices=["y", "n"], default="y")
    if confirm == "n":
        return await configure_models(api_key)

    return sub_models, master_model


def make_status_table(
    all_models: list[dict],
    master_model: dict,
    results_received: dict,
) -> Table:
    table = Table(show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Model", style="cyan", no_wrap=True, min_width=22)
    table.add_column("Role", min_width=16)
    table.add_column("Status", justify="center", width=7)
    table.add_column("Preview", no_wrap=False, max_width=52)
    table.add_column("Time", justify="right", width=7)

    for m in all_models:
        is_master = m["id"] == master_model["id"]
        role = m.get("role", "")

        if is_master:
            role_cell = Text("Master", style="yellow bold")
        elif role:
            role_cell = Text(role, style=ROLES[role]["color"])
        else:
            role_cell = Text("Sub", style="blue")

        r: ModelResult | None = results_received.get(m["id"])
        if r is None:
            status = Text("...", style="dim")
            preview = Text("waiting", style="dim")
            elapsed = ""
        elif r.content:
            status = Text("OK", style="bold green")
            snippet = r.content.strip().replace("\n", " ")
            preview = Text((snippet[:49] + "…") if len(snippet) > 49 else snippet)
            elapsed = f"{r.elapsed_seconds:.1f}s"
        else:
            status = Text("FAIL", style="bold red")
            preview = Text((r.error or "")[:49], style="red")
            elapsed = f"{r.elapsed_seconds:.1f}s"

        table.add_row(m["label"], role_cell, status, preview, elapsed)

    return table


async def run_query(
    prompt: str,
    api_key: str,
    sub_models: list[dict],
    master_model: dict,
    category: str | None = None,
) -> None:
    all_models = sub_models + [master_model]
    results_received: dict[str, ModelResult] = {}

    def on_result_live(result: ModelResult) -> None:
        results_received[result.model_id] = result
        live.update(make_status_table(all_models, master_model, results_received))

    cat_color = CATEGORIES[category]["color"] if category and category in CATEGORIES else "white"
    cat_tag = f"  [{cat_color}][{category}][/]" if category else ""
    console.print(f"\n[bold]Query[/]{cat_tag}: {prompt}\n")

    with Live(
        make_status_table(all_models, master_model, results_received),
        console=console,
        refresh_per_second=4,
    ) as live:
        outcome = await run_synthesis(
            prompt, api_key, sub_models, master_model, on_result=on_result_live, category=category
        )

    # Individual sub-model responses
    console.print("\n[bold magenta]━━━ Individual Responses ━━━[/]\n")
    for r in outcome["sub_results"]:
        role = r.role
        color = ROLES[role]["color"] if role in ROLES else "blue"
        role_tag = f"  [{color}][{role}][/]" if role else ""
        if r.content:
            console.print(Panel(
                Markdown(r.content),
                title=f"[cyan]{r.label}[/]{role_tag}  [dim]{r.elapsed_seconds:.1f}s[/]",
                border_style=color,
                padding=(0, 1),
            ))
        else:
            console.print(Panel(
                f"[red]Failed:[/] {r.error}",
                title=f"[red]{r.label}[/]{role_tag}",
                border_style="red",
            ))

    # Disagreement detection result
    dr = outcome.get("disagreements")
    disagreements_found = (
        dr is not None
        and dr.content is not None
        and not dr.content.strip().upper().startswith(DISAGREEMENT_ABSENT_MARKER)
    )

    # Master's independent answer
    mr = outcome["master_initial"]
    if mr.content:
        console.print(Panel(
            Markdown(mr.content),
            title=f"[yellow]{mr.label} — Independent View[/]  [dim]{mr.elapsed_seconds:.1f}s[/]",
            border_style="yellow",
            padding=(0, 1),
        ))
    else:
        console.print(Panel(
            f"[red]Failed:[/] {mr.error}",
            title=f"[red]{mr.label} — Independent View[/]",
            border_style="red",
        ))

    # Key disagreements panel (Phase 1.5 output)
    console.print()
    if disagreements_found:
        console.print(Panel(
            Markdown(dr.content.strip()),
            title=f"[bold orange3]KEY DISAGREEMENTS[/]  [dim]{dr.elapsed_seconds:.1f}s[/]",
            border_style="orange3",
            padding=(0, 1),
        ))
    elif dr is not None and dr.content:
        console.print("[dim]  ✓ Sub-models broadly agree — no significant disagreements detected.[/]")
    elif dr is not None and dr.error:
        console.print(f"[dim]  Disagreement detection failed: {dr.error}[/]")

    # Final synthesis
    sr = outcome["synthesis"]
    cat_label = f"  [{cat_color}]{category}[/]" if category else ""
    if sr.content:
        console.print(Panel(
            Markdown(sr.content),
            title=f"[bold green]FINAL SYNTHESIZED ANSWER[/]{cat_label}  [dim]{sr.elapsed_seconds:.1f}s[/]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print(f"\n[bold red]Synthesis failed:[/] {sr.error}")


async def main() -> None:
    api_key = get_api_key()

    console.print(Panel.fit(
        "[bold cyan]SynthesizAIr[/]\n"
        "[dim]5 sub-models + 1 master orchestrator via OpenRouter[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    sub_models, master_model = await configure_models(api_key)

    role_summary = "  ".join(
        f"[{ROLES[m['role']]['color']}]{m['role']}[/]" for m in sub_models if m.get("role")
    )
    console.print(
        f"\n[green]Ready.[/] Roles: {role_summary}\n"
        f"Master: [yellow]{master_model['label']}[/]\n"
        "[dim]Type 'quit' to exit.[/]"
    )

    while True:
        try:
            category = select_category()
            if category is None:
                console.print("[dim]Goodbye.[/]")
                break
            prompt = Prompt.ask("\n[bold green]>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/]")
            break

        stripped = prompt.strip()
        if not stripped:
            continue
        if stripped.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        await run_query(stripped, api_key, sub_models, master_model, category=category)


if __name__ == "__main__":
    asyncio.run(main())
