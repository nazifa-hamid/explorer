"""The graph.

                        ┌──────── tools ◀───┐  asked for tools
                        ▼                   │
    START ──▶ agent ──▶ should_continue ────┼── no notes yet ──▶ nudge ──┐
                ▲                           │                            │
                │                           └── done ──▶ finish ──▶ END  │
                └────────────────────────────────────────────────────────┘

Three working nodes and one conditional edge. `should_continue` and `run_tools` are
module-level functions rather than closures on purpose: it means the entire control
flow and every tool side effect can be tested without an LLM, a network, or an API
key. That is the single most useful thing you can do for your own debugging.

Two of the guards here exist because of a real run rather than a design session. The
agent searched twelve times, read two pages, saved zero notes, and produced an empty
report — having obeyed "never state a fact you did not save" perfectly, with nothing
saved. Rewording the prompt helped a little. What fixed it was the graph refusing:
`nudge` will not let a finish through with an empty notes list, and `run_tools`
declines a search when unread results are already in hand. Prompts ask; graphs
enforce.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .models import get_model
from .prompts import (
    ADVICE_NO_NOTES_UNREAD,
    ADVICE_NO_PLAN,
    ADVICE_UNCOVERED,
    ADVICE_UNREAD,
    ADVICE_WRAP_UP,
    COACH_AFTER_FETCH,
    NUDGE_NO_NOTES,
    NUDGE_UNCOVERED,
    SALVAGE_REPORT,
    SEARCH_BLOCKED,
    STATUS,
    STATUS_NO_PLAN,
    STATUS_PLAN,
    STATUS_URGENT,
    SYSTEM,
    WRITE_REPORT,
    format_notes,
    user_prompt,
)
from .state import ResearchState
from .tools import TOOLS, TOOLS_BY_NAME

MAX_STEPS = 12
MAX_NUDGES = 2

# Below this, a repeat search is probably a genuine new angle rather than avoidance.
MIN_UNREAD_TO_BLOCK = 3

URL_RE = re.compile(r"https?://\S+")

# Questions containing these are asking about the present. The model's training data
# is not the present, so its searches need a recency filter whether it asks for one
# or not — left to itself it happily reads three-year-old leaderboards.
RECENCY_WORDS = re.compile(
    r"\b(current|currently|latest|newest|now|today|leading|lead|best|top|state[- ]of[- ]"
    r"the[- ]art|sota|recent|recently|20\d\d)\b",
    re.IGNORECASE,
)
DEFAULT_RECENCY = "y"


def message_text(msg) -> str:
    """Message content as a plain string.

    Most providers return a string, but some return a list of content blocks. Handling
    both here keeps every caller from having to care.
    """
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        content = "".join(parts)
    return content.strip()


def drop_dangling_tool_calls(messages: list) -> list:
    """Remove trailing assistant tool calls that have no matching results.

    Providers reject a history whose last assistant turn requested tools that were
    never answered. That is exactly the state the graph is in when the step budget
    cuts a turn short, so any node that re-sends history has to clean it first.
    """
    answered = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    cleaned = []
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls and not all(c["id"] in answered for c in calls):
            continue
        cleaned.append(msg)
    return cleaned


def extract_urls(text: str) -> list[str]:
    """Pull result URLs out of a web_search result block."""
    return [u.rstrip(".,)") for u in URL_RE.findall(text)]


def default_recency(question: str) -> str | None:
    """A recency filter for questions about the present, if the model omitted one."""
    return DEFAULT_RECENCY if RECENCY_WORDS.search(question) else None


def unread_candidates(state: ResearchState) -> list[str]:
    """URLs a search surfaced that have not been read yet."""
    seen = set(state.get("visited") or [])
    out, dedupe = [], set()
    for url in state.get("candidates") or []:
        if url not in seen and url not in dedupe:
            dedupe.add(url)
            out.append(url)
    return out


def read_since_last_search(messages: list) -> bool:
    """Did a page get fetched after the most recent search?

    Walking backwards is the cheap way to ask "am I searching twice in a row" without
    keeping another counter in state.
    """
    for msg in reversed(messages):
        name = getattr(msg, "name", None)
        if name == "fetch_page":
            return not str(msg.content).startswith(("[error]", "[skipped]"))
        if name == "web_search":
            return False
    return True  # nothing has happened yet


def uncovered(state: ResearchState) -> list[tuple[int, str]]:
    """Sub-questions with no note against them, as (1-based number, text).

    This is the coverage check the agent could not perform on itself. Tagging notes
    with `answers` is what makes it possible, which is why that argument exists on
    save_note rather than a separate 'mark resolved' tool the model would forget.
    """
    plan = state.get("plan") or []
    covered = {n.get("answers") for n in state["notes"] if n.get("answers")}
    return [(i, q) for i, q in enumerate(plan, 1) if i not in covered]


def finish_blocker(state: ResearchState) -> str | None:
    """Why the agent is not allowed to stop yet, or None if it may.

    Two conditions, both learned from failed runs: a report needs notes, and a report
    needs to cover the question that was asked. Returning the reason as text lets the
    nudge node say something specific instead of a generic complaint.
    """
    if not state["notes"]:
        return NUDGE_NO_NOTES

    missing = uncovered(state)
    if missing:
        return NUDGE_UNCOVERED.format(
            missing="\n".join(f"  {i}. {q}" for i, q in missing)
        )
    return None


def status_line(state: ResearchState, max_steps: int) -> str:
    """A compact 'where am I' block, rebuilt every turn.

    The agent has no other way to know its own position. Without this it behaves
    identically on turn 2 and turn 11 — which is exactly what the first real run did,
    searching for new angles with one turn of budget left.

    This changes the system prompt every turn, which defeats prompt caching. On free
    tiers that costs nothing; on a metered model it would be worth moving into a
    trailing message instead.
    """
    turn = state["steps"] + 1
    unread = len(unread_candidates(state))
    notes, read = len(state["notes"]), len(state.get("visited") or [])
    plan = state.get("plan") or []
    missing = uncovered(state)

    if not plan:
        plan_line, advice = STATUS_NO_PLAN, ADVICE_NO_PLAN
    else:
        plan_line = STATUS_PLAN.format(
            done=len(plan) - len(missing),
            total=len(plan),
            missing=(
                "  — still open: " + ", ".join(str(i) for i, _ in missing)
                if missing
                else ""
            ),
        )
        # Order matters: read what you already have, then chase what is missing,
        # then stop. Note there is no "go and search" branch — once a plan exists,
        # "sub-question 2 is still open" is always the more useful way to say it.
        if unread and not notes:
            advice = ADVICE_NO_NOTES_UNREAD
        elif unread:
            advice = ADVICE_UNREAD
        elif missing:
            advice = ADVICE_UNCOVERED.format(n=missing[0][0])
        else:
            advice = ADVICE_WRAP_UP

    return STATUS.format(
        turn=turn,
        max_steps=max_steps,
        urgency=STATUS_URGENT if turn >= max_steps - 2 else "",
        plan=plan_line,
        notes=notes,
        read=read,
        unread=unread,
        advice=advice,
    )


def should_continue(
    state: ResearchState, max_steps: int = MAX_STEPS, max_nudges: int = MAX_NUDGES
) -> str:
    """The entire control flow of a single agent."""
    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        return "finish" if state["steps"] >= max_steps else "tools"

    # The model wants to stop. Let it, unless it has nothing to write a report from,
    # or has not covered its own plan, and there is still budget to fix that.
    out_of_budget = state["steps"] >= max_steps
    nudged_enough = state.get("nudges", 0) >= max_nudges
    if finish_blocker(state) and not out_of_budget and not nudged_enough:
        return "nudge"

    return "finish"


def _coach(name: str, args: dict, result: str) -> str:
    """Append a just-in-time reminder to a tool result.

    Coaching rides along with tool output rather than sitting in the system prompt,
    because it lands at the moment it is relevant instead of a dozen turns earlier.

    Only one reminder survives. A second, warning about repeated searching with no
    notes to show, was made redundant twice over: the search block refuses that move
    outright, and the status line reports the note count every turn regardless.
    """
    if name == "fetch_page" and not result.startswith(("[error]", "[skipped]")):
        return result + COACH_AFTER_FETCH.format(url=args.get("url", ""))
    return result


def run_tools(state: ResearchState) -> dict:
    """Execute every tool the model asked for and fold the results back into state."""
    last = state["messages"][-1]
    out_msgs: list[ToolMessage] = []
    notes, visited, candidates = [], [], []
    plan: list[str] | None = None  # None means "not touched"; [] would wipe it

    # The prompt asks the model not to refetch, but a prompt is a request, not a
    # guarantee. Enforcing it in the node saves real tokens and latency.
    already_visited = set(state.get("visited") or [])
    unread = unread_candidates(state)
    just_read = read_since_last_search(state["messages"])

    for call in last.tool_calls:
        name, args = call["name"], call["args"]
        tool = TOOLS_BY_NAME.get(name)

        if tool is None:
            result = f"[error] unknown tool {name!r}"
        elif name == "fetch_page" and args.get("url", "") in already_visited:
            result = (
                f"[skipped] {args['url']} was already fetched earlier in this run; "
                "use the notes you saved from it instead of refetching"
            )
        elif (
            name == "web_search"
            and not just_read
            and len(unread) >= MIN_UNREAD_TO_BLOCK
        ):
            # The core fix for searching every turn. One search yields ~5 URLs;
            # searching again without opening any of them is avoidance, not research.
            # Refusing is what stops it — a reminder only asks.
            result = SEARCH_BLOCKED.format(
                n=len(unread), urls="\n".join(f"- {u}" for u in unread[:6])
            )
        else:
            # A question about "the leading X" needs recent sources. The model
            # rarely sets this itself, so fill it in rather than let it read
            # whatever ranks highest, which skews old.
            if name == "web_search" and not args.get("recency"):
                inferred = default_recency(state["question"])
                if inferred:
                    args = {**args, "recency": inferred}

            try:
                result = tool.invoke(args)
            except Exception as e:  # a bad tool call must never kill the graph
                result = f"[error] {name} failed: {e}"

            # State side effects live here, not inside the tools.
            if name == "save_note":
                notes.append(
                    {
                        "claim": args.get("claim", ""),
                        "source_url": args.get("source_url", ""),
                        "answers": args.get("answers"),
                    }
                )
            elif name == "set_plan":
                plan = [str(q) for q in args.get("sub_questions", []) if str(q).strip()]
            elif name == "fetch_page":
                url = args.get("url", "")
                visited.append(url)
                already_visited.add(url)  # dedupe within this batch too
            elif name == "web_search":
                candidates.extend(extract_urls(str(result)))

        result = _coach(name, args, str(result))
        out_msgs.append(
            ToolMessage(content=result, tool_call_id=call["id"], name=name)
        )

    update = {
        "messages": out_msgs,
        "notes": notes,
        "visited": visited,
        "candidates": candidates,
    }
    if plan is not None:  # omit the key entirely so a no-op cannot clear the plan
        update["plan"] = plan
    return update


def nudge(state: ResearchState) -> dict:
    """Push back when the agent tries to finish prematurely, saying exactly why."""
    return {
        "messages": [HumanMessage(finish_blocker(state) or NUDGE_NO_NOTES)],
        "nudges": state.get("nudges", 0) + 1,
    }


def build_graph(
    model_name: str = "glm",
    max_steps: int = MAX_STEPS,
    checkpointer=None,
    max_nudges: int = MAX_NUDGES,
):
    """Compile the graph. `model_name` is looked up in the registry, never imported.

    Retries are the client's job, not ours — see MAX_RETRIES in models.py. An earlier
    version wrapped these in `with_retry` as a second net, which by default retries
    *every* exception: a 404 for a bad model name and an exhausted daily quota, both
    permanent, were retried until they took two minutes to report.
    """
    llm = get_model(model_name).bind_tools(TOOLS)
    plain_llm = get_model(model_name)  # no tools — used to force a report

    def agent(state: ResearchState) -> dict:
        # The status block is appended to the system prompt rather than pushed into
        # history: it is rebuilt fresh each turn and never accumulates.
        system = SystemMessage(SYSTEM + "\n\n" + status_line(state, max_steps))
        reply = llm.invoke([system] + state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def finish(state: ResearchState) -> dict:
        """Produce the report.

        Normally this is just the prose the model already wrote, because it writes
        that with the pages it read still in context and the result is richer than
        anything rebuilt from notes alone.

        We tried the stricter alternative — always rebuild from the saved notes, so
        that an unsaved fact literally cannot appear in the report. It worked, and the
        reports got much thinner, because notes are lossy summaries of what was read.
        The strict version buys honest citations at the cost of most of the substance,
        which was a bad trade here. Worth revisiting once a critic node can check
        claims against notes instead of the writer being starved of them.
        """
        last = state["messages"][-1]
        text = message_text(last)
        if not getattr(last, "tool_calls", None) and text:
            return {"report": text}

        if state["notes"]:
            prompt = WRITE_REPORT.format(
                question=state["question"],
                notes=format_notes(state["notes"], state.get("plan")),
            )
            reply = plain_llm.invoke([SystemMessage(SYSTEM), HumanMessage(prompt)])
        else:
            # No notes and no room left to nudge. Salvage something from whatever was
            # read rather than returning an empty report — degraded, but not useless.
            history = drop_dangling_tool_calls(state["messages"])
            reply = plain_llm.invoke(
                [SystemMessage(SYSTEM), *history,
                 HumanMessage(SALVAGE_REPORT.format(question=state["question"]))]
            )

        return {"messages": [reply], "report": message_text(reply)}

    g = StateGraph(ResearchState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("nudge", nudge)
    g.add_node("finish", finish)

    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        lambda s: should_continue(s, max_steps, max_nudges),
        {"tools": "tools", "nudge": "nudge", "finish": "finish"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("nudge", "agent")
    g.add_edge("finish", END)

    return g.compile(checkpointer=checkpointer)


def initial_state(question: str) -> ResearchState:
    return {
        "question": question,
        "plan": [],
        "messages": [HumanMessage(user_prompt(question))],
        "notes": [],
        "visited": [],
        "candidates": [],
        "steps": 0,
        "nudges": 0,
        "report": None,
    }


__all__ = [
    "MAX_NUDGES",
    "MAX_STEPS",
    "MIN_UNREAD_TO_BLOCK",
    "build_graph",
    "default_recency",
    "drop_dangling_tool_calls",
    "extract_urls",
    "finish_blocker",
    "initial_state",
    "message_text",
    "nudge",
    "read_since_last_search",
    "run_tools",
    "should_continue",
    "status_line",
    "uncovered",
    "unread_candidates",
]
