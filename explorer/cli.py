"""The trace is the deliverable.

Should be able to read a run top to bottom and see exactly why the agent did what
it did: which query it chose, which page it read, which fact it kept. If a run goes
wrong and the trace does not show why, trace needs to improve first.
"""

import json
import pathlib
import time
import uuid

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .graph import MAX_STEPS, build_graph, initial_state, message_text

app = typer.Typer(add_completion=False, help="A single research agent on LangGraph.")
console = Console()

RUNS = pathlib.Path("runs")
DB = RUNS / "runs.sqlite"
PREVIEW_CHARS = 160


def _render(node: str, update: dict) -> None:
    """Print one node's update. This is the whole observability layer."""
    if node == "agent":
        msg = update["messages"][-1]
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                console.print(
                    f"  [bold cyan]→ {call['name']}[/bold cyan]({escape(args)})"
                )
        else:
            text = message_text(msg)
            if text:
                console.print(f"  [dim]agent wrote {len(text)} chars[/dim]")

    elif node == "nudge":
        reason = " ".join(str(update["messages"][0].content).split())[:120]
        console.print(f"  [magenta]⟲ nudge[/magenta] [dim]{escape(reason)}…[/dim]")

    elif node == "tools":
        for msg in update["messages"]:
            preview = " ".join(str(msg.content).split())[:PREVIEW_CHARS]
            if preview.startswith("[error]"):
                colour = "red"
            elif preview.startswith("[skipped]"):
                colour = "yellow"
            else:
                colour = "green"
            # escape() matters here: tool output is full of [skipped], [error] and
            # [1] markers, and Rich would read those as style tags and silently eat
            # them — quietly corrupting the one artifact this tool exists to produce.
            console.print(
                f"  [{colour}]← {msg.name}[/{colour}] [dim]{escape(preview)}…[/dim]"
            )


def _is_rate_limit(message: str) -> bool:
    lowered = message.lower()
    return "429" in lowered or "rate limit" in lowered or "overloaded" in lowered


def _is_daily_quota(message: str) -> bool:
    """A per-day cap, as opposed to a per-minute burst limit.

    The distinction matters and the APIs obscure it: Google returns a daily-quota
    rejection carrying `retryDelay: 58s`, which invites you to wait a minute and try
    again forever. Retrying does not help; the quota is per model per day, so the
    only moves are a different model or tomorrow.
    """
    lowered = message.lower()
    return "resource_exhausted" in lowered or "perday" in lowered.replace("_", "")


def _write_report(
    path: pathlib.Path,
    question: str,
    report: str,
    notes: list,
    failure: str | None = None,
    plan: list | None = None,
) -> None:
    lines = [f"# {question}", "", report or "_(no report produced)_", ""]
    if failure:
        lines += [f"> Run ended early: `{failure}`", ""]
    if plan:
        covered = {n.get("answers") for n in notes if n.get("answers")}
        lines += ["---", "", "## Plan", ""]
        lines += [
            f"{i}. {'✓' if i in covered else '✗'} {q}"
            for i, q in enumerate(plan, 1)
        ]
        lines += [""]
    if notes:
        lines += ["---", "", "## Notes gathered", ""]
        lines += [f"- {n['claim']} — <{n['source_url']}>" for n in notes]
        lines += [""]
    path.write_text("\n".join(lines))


@app.command()
def research(
    question: str = typer.Argument(..., help="The research question."),
    model: str = typer.Option("glm", help="glm | local | gemini"),
    max_steps: int = typer.Option(MAX_STEPS, help="Hard cap on agent turns."),
) -> None:
    """Research a question and write a cited report."""
    RUNS.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    console.rule(f"[bold]{question}[/bold]")
    console.print(f"[dim]model={model}  max_steps={max_steps}  run={run_id}[/dim]\n")

    report: str | None = None
    notes: list = []
    visited: list = []
    plan: list = []
    turns = 0
    nudges = 0
    failure: str | None = None
    started = time.monotonic()

    try:
        with SqliteSaver.from_conn_string(str(DB)) as saver:
            graph = build_graph(model, max_steps, checkpointer=saver)
            config = {"configurable": {"thread_id": run_id}}

            for chunk in graph.stream(
                initial_state(question), config, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    if node == "agent":
                        turns += 1
                        console.print(f"[bold]turn {turns}[/bold]")
                    _render(node, update)
                    if node == "nudge":
                        nudges = update.get("nudges", nudges)
                    if update.get("plan"):
                        plan = update["plan"]
                        console.print("  [bold]plan:[/bold]")
                        for i, q in enumerate(plan, 1):
                            console.print(f"    [dim]{i}.[/dim] {escape(q)}")
                    notes.extend(update.get("notes") or [])
                    visited.extend(update.get("visited") or [])
                    if update.get("report"):
                        report = update["report"]
    except KeyboardInterrupt:
        failure = "interrupted"
        console.print("\n[yellow]interrupted[/yellow]")
    except Exception as e:
        # A failed call should not throw away the notes already gathered — they are
        # the expensive part. Report what we have and say what went wrong.
        failure = str(e)
        console.print(f"\n[red]run failed after {turns} turns:[/red] {escape(failure)}")
        if _is_daily_quota(failure):
            console.print(
                "[yellow]That is a per-day quota, not a burst limit.[/yellow] "
                "Retrying will not help, and the retryDelay in that error is "
                "misleading. The quota is per model, so a different model gets a "
                "fresh allowance: set GEMINI_MODEL in .env, or use --model glm / "
                "--model local. Otherwise it resets at midnight Pacific."
            )
        elif _is_rate_limit(failure):
            console.print(
                "[yellow]That is a rate limit, not a bug.[/yellow] The free tier "
                "throttles bursts. Wait a minute and retry, lower --max-steps, or "
                "switch backend with --model local / --model glm."
            )

    elapsed = time.monotonic() - started

    console.rule("report")
    console.print(Panel(Markdown(report or "_(no report produced)_")))

    covered = len({n.get("answers") for n in notes if n.get("answers")})
    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]turns[/dim]", str(turns))
    summary.add_row("[dim]notes[/dim]", str(len(notes)))
    summary.add_row("[dim]pages fetched[/dim]", str(len(visited)))
    summary.add_row(
        "[dim]distinct sources[/dim]",
        str(len({n.get("source_url") for n in notes if n.get("source_url")})),
    )
    if plan:
        summary.add_row("[dim]plan covered[/dim]", f"{covered} of {len(plan)}")
    summary.add_row("[dim]nudges[/dim]", str(nudges))
    summary.add_row("[dim]elapsed[/dim]", f"{elapsed:.1f}s")
    console.print(summary)

    sources = {n.get("source_url") for n in notes if n.get("source_url")}
    if len(sources) == 1:
        console.print(
            "\n[yellow]Every note came from one source.[/yellow] The report may be "
            "accurate and still one-sided — check whether that source has an interest "
            "in the answer."
        )

    if plan and covered < len(plan):
        console.print(
            f"\n[yellow]{len(plan) - covered} sub-question(s) went unanswered[/yellow] "
            "— the report may have gaps the agent did not flag."
        )

    if not notes:
        console.print(
            "\n[yellow]No notes were saved,[/yellow] so the report is unciteable. "
            "Check the trace: did the agent read any pages at all?"
        )

    out = RUNS / f"{run_id}.md"
    _write_report(out, question, report or "", notes, failure, plan)
    console.print(f"\n[dim]report → {out}[/dim]")
    console.print(f"[dim]replay → run-explorer replay {run_id}[/dim]")

    if failure:
        raise typer.Exit(1)


@app.command()
def replay(run_id: str = typer.Argument(..., help="Run id printed by `research`.")) -> None:
    """Replay a saved run's messages from its checkpoint."""
    if not DB.exists():
        console.print(f"[red]no checkpoint database at {DB}[/red]")
        raise typer.Exit(1)

    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})

    if not snapshot.values:
        console.print(f"[red]no run found with id {run_id}[/red]")
        raise typer.Exit(1)

    console.rule(f"[bold]{snapshot.values.get('question', run_id)}[/bold]")
    for msg in snapshot.values.get("messages", []):
        kind = msg.__class__.__name__.replace("Message", "").lower()
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                console.print(
                    f"[bold cyan]→ {call['name']}[/bold cyan] {escape(str(call['args']))}"
                )
        else:
            text = message_text(msg)
            console.print(f"[dim]{kind}:[/dim] {escape(text[:400])}")

    notes = snapshot.values.get("notes", [])
    console.print(f"\n[dim]{len(notes)} notes[/dim]")
    for note in notes:
        console.print(f"  • {note['claim']} [dim]{note['source_url']}[/dim]")


@app.command("show-graph")
def show_graph() -> None:
    """Print the graph as Mermaid, to paste into a README or mermaid.live."""
    graph = build_graph()
    console.print(graph.get_graph().draw_mermaid())


@app.command("models")
def models() -> None:
    """List the Gemini models your key can actually use, with their real IDs.

    Model IDs are not guessable ("gemini-3-flash" does not exist; "gemini-3.6-flash"
    does) and aliases like -latest resolve to whatever is newest, which tends to have
    the smallest free-tier quota. This asks the API instead of guessing. Pick one and
    set GEMINI_MODEL in .env.
    """
    import os

    import httpx
    from dotenv import dotenv_values

    key = os.getenv("GOOGLE_API_KEY") or dotenv_values(".env").get("GOOGLE_API_KEY")
    if not key:
        console.print("[red]no GOOGLE_API_KEY in the environment or .env[/red]")
        raise typer.Exit(1)

    resp = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": key, "pageSize": 200},
        timeout=30,
    )
    resp.raise_for_status()

    table = Table(box=None, pad_edge=False)
    table.add_column("model id", style="cyan")
    table.add_column("display name", style="dim")
    for m in resp.json().get("models", []):
        name = m["name"].removeprefix("models/")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if any(x in name for x in ("image", "tts", "live", "embed", "veo", "audio")):
            continue
        table.add_row(name, m.get("displayName", ""))
    console.print(table)
    console.print(
        "\n[dim]Set GEMINI_MODEL in .env to one of these. Flash models have the "
        "larger free quotas; check https://ai.google.dev/gemini-api/docs/rate-limits"
        "[/dim]"
    )


@app.command("check")
def check(model: str = typer.Option("glm", help="glm | local | gemini")) -> None:
    """Verify a model can emit a tool call AND accept the result back.

    Two turns, not one, and the second turn is the point. A single call proves very
    little: providers that mangle tool-call history only fail when you replay it.
    Gemini's OpenAI-compatible endpoint passes turn 1 and then 400s on turn 2 because
    the `thought_signature` attached to its tool calls does not survive the round
    trip. Checking one turn would have declared that backend healthy.
    """
    from langchain_core.messages import HumanMessage, ToolMessage

    from .models import get_model
    from .tools import TOOLS

    console.print(f"[dim]checking {model}…[/dim]")

    # --- turn 1: can it emit a tool call? ---
    try:
        llm = get_model(model).bind_tools(TOOLS)
        history = [HumanMessage("Search the web for the price of tea in China.")]
        reply = llm.invoke(history)
    except Exception as e:
        console.print(f"[red]✗ could not reach {model}:[/red] {e}")
        raise typer.Exit(1) from e

    if not getattr(reply, "tool_calls", None):
        console.print("[red]✗ no tool_calls in the response[/red]")
        console.print(
            "The model replied with prose instead of a tool call:\n"
            f"[dim]{message_text(reply)[:300]}[/dim]\n\n"
            "For --model local this usually means llama-server was started without "
            "--jinja, so the tool-call chat template is not being applied."
        )
        raise typer.Exit(1)

    console.print("[green]✓ turn 1: emits tool calls[/green]")
    console.print(json.dumps(reply.tool_calls, indent=2, default=str))

    # --- turn 2: can it accept the result back? ---
    history.append(reply)
    for call in reply.tool_calls:
        history.append(
            ToolMessage(
                content="[1] Tea Prices\nhttp://example.com\nAround $12/kg.",
                tool_call_id=call["id"],
                name=call["name"],
            )
        )

    try:
        second = llm.invoke(history)
    except Exception as e:
        console.print(f"[red]✗ turn 2 failed replaying tool results:[/red] {e}")
        if "thought_signature" in str(e):
            console.print(
                "[yellow]This backend drops Gemini's thought_signature.[/yellow] Use "
                "the native client (langchain-google-genai) rather than the "
                "OpenAI-compatible endpoint."
            )
        raise typer.Exit(1) from e

    console.print("[green]✓ turn 2: accepts tool results back[/green]")
    preview = message_text(second)[:200] or "(another tool call)"
    console.print(f"[dim]{preview}[/dim]")
    console.print("\n[bold green]tool loop round-trips — good to run[/bold green]")


if __name__ == "__main__":
    app()
