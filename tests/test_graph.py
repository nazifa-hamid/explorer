"""Routing and tool-node behaviour, tested with no LLM, no network, no API key.

This is possible only because `should_continue` and `run_tools` are plain functions
that take state and return state. Keeping them out of closures is what makes the
control flow testable at all.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from explorer.graph import (
    default_recency,
    drop_dangling_tool_calls,
    extract_urls,
    finish_blocker,
    message_text,
    nudge,
    read_since_last_search,
    run_tools,
    should_continue,
    status_line,
    uncovered,
    unread_candidates,
)


def _tool_call(name: str, args: dict, call_id: str = "1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _state(
    last,
    steps: int = 0,
    visited: list | None = None,
    notes: list | None = None,
    nudges: int = 0,
    plan: list | None = None,
) -> dict:
    return {
        "question": "q",
        "plan": plan if plan is not None else [],
        "messages": [last],
        "notes": notes if notes is not None else [],
        "visited": visited or [],
        "candidates": [],
        "steps": steps,
        "nudges": nudges,
        "report": None,
    }


NOTE = {"claim": "c", "source_url": "http://a", "answers": 1}
PLAN = ["what does it cost?", "what are the limits?"]

# A plan whose single sub-question the NOTE above covers, so tests that are not
# about coverage do not accidentally trip the coverage guard.
COVERED = {"plan": ["only question"], "notes": [NOTE]}


# --- routing ---------------------------------------------------------------


def test_no_tool_calls_with_a_covered_plan_goes_to_finish():
    state = _state(AIMessage("done"), **COVERED)
    assert should_continue(state) == "finish"


def test_no_tool_calls_without_notes_gets_nudged():
    """The failure that motivated the nudge node: stopping with nothing to report."""
    assert should_continue(_state(AIMessage("done"))) == "nudge"


def test_finishing_with_an_uncovered_sub_question_gets_nudged():
    """The coverage guard: notes exist, but not for everything the plan asked."""
    state = _state(AIMessage("done"), plan=PLAN, notes=[NOTE])  # covers only #1
    assert should_continue(state) == "nudge"


def test_nudging_stops_after_the_cap():
    state = _state(AIMessage("done"), nudges=2)
    assert should_continue(state, max_nudges=2) == "finish"


def test_no_budget_left_means_no_nudge():
    """Nudging costs a turn; do not spend one that isn't there."""
    state = _state(AIMessage("done"), steps=12)
    assert should_continue(state, max_steps=12) == "finish"


def test_tool_calls_go_to_tools():
    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "x"})])
    assert should_continue(_state(msg)) == "tools"


def test_budget_exhausted_forces_finish():
    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "x"})])
    assert should_continue(_state(msg, steps=12), max_steps=12) == "finish"


def test_budget_not_yet_reached_keeps_going():
    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "x"})])
    assert should_continue(_state(msg, steps=11), max_steps=12) == "tools"


# --- tool node -------------------------------------------------------------


def test_save_note_lands_in_state():
    msg = AIMessage(
        "",
        tool_calls=[
            _tool_call(
                "save_note",
                {"claim": "Gemma 4 scores 86.4 on tau2-bench", "source_url": "http://x"},
            )
        ],
    )
    out = run_tools(_state(msg))
    assert out["notes"] == [
        {
            "claim": "Gemma 4 scores 86.4 on tau2-bench",
            "source_url": "http://x",
            "answers": None,
        }
    ]
    assert out["messages"][0].name == "save_note"
    assert "plan" not in out  # a turn that never touched the plan must not clear it


def test_unknown_tool_does_not_crash():
    msg = AIMessage("", tool_calls=[_tool_call("nope", {})])
    out = run_tools(_state(msg))
    assert "[error]" in out["messages"][0].content
    assert out["notes"] == []


def test_refetching_a_visited_url_is_skipped(monkeypatch):
    called = []
    monkeypatch.setattr(
        "explorer.tools.httpx.get", lambda *a, **k: called.append(1)
    )

    msg = AIMessage("", tool_calls=[_tool_call("fetch_page", {"url": "http://a"})])
    out = run_tools(_state(msg, visited=["http://a"]))

    assert "[skipped]" in out["messages"][0].content
    assert called == []  # the network was never touched
    assert out["visited"] == []  # and it was not recorded twice


def test_two_tool_calls_in_one_turn_both_execute():
    msg = AIMessage(
        "",
        tool_calls=[
            _tool_call("save_note", {"claim": "a", "source_url": "http://a"}, "1"),
            _tool_call("save_note", {"claim": "b", "source_url": "http://b"}, "2"),
        ],
    )
    out = run_tools(_state(msg))
    assert len(out["messages"]) == 2
    assert [n["claim"] for n in out["notes"]] == ["a", "b"]


def test_tool_message_ids_match_the_calls():
    """Mismatched tool_call_ids are rejected by most providers, so this matters."""
    msg = AIMessage(
        "",
        tool_calls=[
            _tool_call("save_note", {"claim": "a", "source_url": "http://a"}, "abc")
        ],
    )
    out = run_tools(_state(msg))
    assert out["messages"][0].tool_call_id == "abc"


# --- nudge -----------------------------------------------------------------


def test_nudge_appends_an_instruction_and_counts():
    out = nudge(_state(AIMessage("done")))
    assert out["nudges"] == 1
    assert "save_note" in out["messages"][0].content


def test_nudge_names_the_uncovered_sub_questions():
    """A generic complaint is useless; the pushback has to say what is missing."""
    state = _state(AIMessage("done"), plan=PLAN, notes=[NOTE])  # #2 uncovered
    out = nudge(state)
    assert "what are the limits?" in out["messages"][0].content
    assert "what does it cost?" not in out["messages"][0].content


# --- the plan tool ---------------------------------------------------------


def test_set_plan_lands_in_state():
    msg = AIMessage(
        "", tool_calls=[_tool_call("set_plan", {"sub_questions": ["a?", "b?", " "]})]
    )
    out = run_tools(_state(msg))
    assert out["plan"] == ["a?", "b?"]  # blank entries dropped


def test_uncovered_tracks_notes_tagged_with_answers():
    state = _state(AIMessage("x"), plan=["a", "b", "c"], notes=[NOTE])  # tagged 1
    assert uncovered(state) == [(2, "b"), (3, "c")]


def test_uncovered_is_empty_with_no_plan():
    """No plan means no coverage claim — the guard must not fire spuriously."""
    assert uncovered(_state(AIMessage("x"))) == []


def test_finish_blocker_prioritises_missing_notes_over_coverage():
    state = _state(AIMessage("x"), plan=PLAN)
    assert "not saved any notes" in finish_blocker(state)


def test_finish_blocker_is_none_when_everything_is_covered():
    assert finish_blocker(_state(AIMessage("x"), **COVERED)) is None


def test_fetching_a_page_appends_a_save_note_reminder(monkeypatch):
    class _Resp:
        text = "<html><body><article>" + ("Real content here. " * 30) + "</article></body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("explorer.tools.httpx.get", lambda *a, **k: _Resp())
    msg = AIMessage("", tool_calls=[_tool_call("fetch_page", {"url": "http://a"})])
    out = run_tools(_state(msg))
    assert "[reminder]" in out["messages"][0].content
    assert "save_note" in out["messages"][0].content


# --- the search-loop fix ---------------------------------------------------


def test_extract_urls_from_a_search_result():
    block = "[1] Title\nhttps://a.com/x\nsnippet\n\n[2] Other\nhttps://b.com/y.\nsnip"
    assert extract_urls(block) == ["https://a.com/x", "https://b.com/y"]


def test_unread_candidates_excludes_visited_and_dupes():
    state = _state(AIMessage("x"))
    state["candidates"] = ["http://a", "http://b", "http://a", "http://c"]
    state["visited"] = ["http://b"]
    assert unread_candidates(state) == ["http://a", "http://c"]


def test_read_since_last_search_detects_search_twice_in_a_row():
    msgs = [
        ToolMessage(content="r", tool_call_id="1", name="web_search"),
        ToolMessage(content="page text", tool_call_id="2", name="fetch_page"),
        ToolMessage(content="r", tool_call_id="3", name="web_search"),
    ]
    assert read_since_last_search(msgs) is False
    assert read_since_last_search(msgs[:2]) is True


def test_a_failed_fetch_does_not_count_as_reading():
    msgs = [
        ToolMessage(content="r", tool_call_id="1", name="web_search"),
        ToolMessage(content="[error] could not fetch", tool_call_id="2", name="fetch_page"),
    ]
    assert read_since_last_search(msgs) is False


def test_searching_again_with_unread_results_is_refused(monkeypatch):
    """The actual fix for searching every turn: the graph refuses, it does not ask."""
    ran = []
    monkeypatch.setattr(
        "explorer.tools.DDGS", lambda: ran.append(1)  # would explode if called
    )

    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "again"})])
    state = _state(msg)
    state["candidates"] = ["http://a", "http://b", "http://c", "http://d"]
    state["messages"] = [
        ToolMessage(content="r", tool_call_id="0", name="web_search"),
        msg,
    ]

    out = run_tools(state)

    assert ran == []  # no search was performed
    assert "[skipped]" in out["messages"][0].content
    assert "http://a" in out["messages"][0].content  # tells it what to read instead


def test_searching_is_allowed_after_reading_something(monkeypatch):
    class _DDGS:
        def text(self, query, **kwargs):
            return [{"title": "T", "href": "http://new", "body": "b"}]

    monkeypatch.setattr("explorer.tools.DDGS", _DDGS)
    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "next angle"})])
    state = _state(msg)
    state["candidates"] = ["http://a", "http://b", "http://c", "http://d"]
    state["messages"] = [
        ToolMessage(content="r", tool_call_id="0", name="web_search"),
        ToolMessage(content="real page text", tool_call_id="1", name="fetch_page"),
        msg,
    ]

    out = run_tools(state)

    assert "[skipped]" not in out["messages"][0].content
    assert out["candidates"] == ["http://new"]


def test_few_unread_results_do_not_block_a_new_angle(monkeypatch):
    class _DDGS:
        def text(self, query, **kwargs):
            return [{"title": "T", "href": "http://new", "body": "b"}]

    monkeypatch.setattr("explorer.tools.DDGS", _DDGS)
    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "x"})])
    state = _state(msg)
    state["candidates"] = ["http://a"]  # below MIN_UNREAD_TO_BLOCK
    state["messages"] = [
        ToolMessage(content="r", tool_call_id="0", name="web_search"),
        msg,
    ]
    assert "[skipped]" not in run_tools(state)["messages"][0].content


# --- recency ---------------------------------------------------------------


def test_recency_inferred_for_present_tense_questions():
    assert default_recency("which AI models are leading in reasoning?") == "y"
    assert default_recency("what is the current best free LLM API?") == "y"
    assert default_recency("free LLM API tiers in 2026") == "y"


def test_no_recency_for_timeless_questions():
    assert default_recency("how does the transformer attention mechanism work?") is None
    assert default_recency("who wrote the Gettysburg Address?") is None


def test_recency_is_applied_to_searches_when_the_model_omits_it(monkeypatch):
    seen = {}

    class _DDGS:
        def text(self, query, **kwargs):
            seen.update(kwargs)
            return []

    monkeypatch.setattr("explorer.tools.DDGS", _DDGS)
    msg = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "x"})])
    state = _state(msg)
    state["question"] = "which AI models are leading right now?"
    run_tools(state)
    assert seen["timelimit"] == "y"


def test_the_models_own_recency_choice_wins(monkeypatch):
    seen = {}

    class _DDGS:
        def text(self, query, **kwargs):
            seen.update(kwargs)
            return []

    monkeypatch.setattr("explorer.tools.DDGS", _DDGS)
    msg = AIMessage(
        "", tool_calls=[_tool_call("web_search", {"query": "x", "recency": "w"})]
    )
    state = _state(msg)
    state["question"] = "what is the latest model?"
    run_tools(state)
    assert seen["timelimit"] == "w"


# --- status line -----------------------------------------------------------


def test_status_reports_position_and_points_at_unread():
    state = _state(AIMessage("x"), steps=3, plan=PLAN)
    state["candidates"] = ["http://a", "http://b"]
    line = status_line(state, max_steps=12)
    assert "turn 4 of 12" in line
    assert "unread search results waiting: 2" in line
    assert "fetch_page" in line  # no notes yet, so it should say to read


def test_status_gets_urgent_near_the_budget():
    state = _state(AIMessage("x"), steps=10, **COVERED)
    line = status_line(state, max_steps=12)
    assert "running out of turns" in line
    assert "Write the report" in line


def test_status_nags_when_no_plan_is_set():
    """A planless agent gets told to plan before anything else."""
    line = status_line(_state(AIMessage("x")), max_steps=12)
    assert "plan: NOT SET" in line
    assert "set_plan" in line


def test_status_shows_plan_coverage():
    state = _state(AIMessage("x"), plan=PLAN, notes=[NOTE])  # covers #1 only
    line = status_line(state, max_steps=12)
    assert "plan: 1 of 2 sub-questions answered" in line
    assert "still open: 2" in line
    assert "research sub-question 2" in line


def test_status_points_a_fresh_planned_agent_at_sub_question_one():
    """With a plan, 'nothing read yet' is better said as 'sub-question 1 is open'."""
    line = status_line(_state(AIMessage("x"), plan=PLAN), max_steps=12)
    assert "research sub-question 1" in line


# --- history hygiene -------------------------------------------------------


def test_drop_dangling_tool_calls_removes_unanswered_requests():
    asked = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "x"}, "1")])
    answered = AIMessage("", tool_calls=[_tool_call("web_search", {"query": "y"}, "2")])
    result = ToolMessage(content="r", tool_call_id="2", name="web_search")
    history = [HumanMessage("q"), answered, result, asked]

    cleaned = drop_dangling_tool_calls(history)

    assert asked not in cleaned
    assert answered in cleaned and result in cleaned


# --- helpers ---------------------------------------------------------------


def test_message_text_handles_content_blocks():
    msg = AIMessage(content=[{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}])
    assert message_text(msg) == "hello world"


def test_message_text_handles_plain_string():
    assert message_text(AIMessage("  spaced  ")) == "spaced"
