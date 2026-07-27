"""Run the whole graph against a scripted fake model.

No API key, no network, no LLM — but every edge in the graph is exercised, including
the budget-exhaustion fallback that is otherwise annoying to trigger on purpose. If
you change the graph and this still passes, the wiring is intact.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage

from explorer import graph as graph_mod
from explorer import tools as tools_mod


class FakeModel:
    """Returns a scripted list of replies, then repeats the last one forever.

    Each reply is copied with a fresh id, and that detail is load-bearing:
    `add_messages` deduplicates by message id, so handing back the *same* message
    object twice replaces the earlier copy in history instead of appending to it.
    Real providers mint a new id per response; a fake that doesn't will produce
    baffling test failures.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen: list = []  # every prompt it was handed, for assertions

    def bind_tools(self, tools):  # the graph calls this; nothing to do
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply.model_copy(update={"id": str(uuid.uuid4())})


def _call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def no_network(monkeypatch):
    class _DDGS:
        def text(self, query, max_results=5):
            return [{"title": "Docs", "href": "http://example.com/a", "body": "snippet"}]

    class _Resp:
        text = "<html><body><article>The answer is forty two, per the docs.</article></body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tools_mod, "DDGS", _DDGS)
    monkeypatch.setattr(tools_mod.httpx, "get", lambda *a, **k: _Resp())


def _install(monkeypatch, model):
    monkeypatch.setattr(graph_mod, "get_model", lambda name: model)


def test_happy_path_plan_search_fetch_note_report(monkeypatch, no_network):
    """plan → search → fetch → tagged note → report, with no pushback along the way."""
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[_call("set_plan", {"sub_questions": ["what is it?"]}, "0")],
            ),
            AIMessage("", tool_calls=[_call("web_search", {"query": "forty two"}, "1")]),
            AIMessage(
                "", tool_calls=[_call("fetch_page", {"url": "http://example.com/a"}, "2")]
            ),
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "save_note",
                        {
                            "claim": "The answer is 42",
                            "source_url": "http://example.com/a",
                            "answers": 1,
                        },
                        "3",
                    )
                ],
            ),
            AIMessage("The answer is 42 [source: http://example.com/a]"),
        ]
    )
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm", max_steps=12)
    final = app.invoke(graph_mod.initial_state("what is the answer?"))

    assert final["report"].startswith("The answer is 42")
    assert final["plan"] == ["what is it?"]
    assert final["notes"] == [
        {
            "claim": "The answer is 42",
            "source_url": "http://example.com/a",
            "answers": 1,
        }
    ]
    assert final["visited"] == ["http://example.com/a"]
    assert final["nudges"] == 0  # the plan was covered, so nothing pushed back
    assert final["steps"] == 5


def test_finishing_with_an_uncovered_sub_question_is_pushed_back(monkeypatch, no_network):
    """The coverage guard end to end: two sub-questions, notes for only one."""
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[_call("set_plan", {"sub_questions": ["cost?", "limits?"]}, "0")],
            ),
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "save_note",
                        {"claim": "it is free", "source_url": "http://a", "answers": 1},
                        "1",
                    )
                ],
            ),
            AIMessage("Done."),  # tries to stop with sub-question 2 still empty
        ]
    )
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm", max_nudges=1)
    final = app.invoke(graph_mod.initial_state("what does it cost and what are limits?"))

    assert final["nudges"] == 1
    pushback = [m for m in final["messages"] if "limits?" in str(m.content)]
    assert pushback, "the nudge should name the uncovered sub-question"
    assert final["report"] == "Done."  # accepted once the nudge budget ran out


def test_the_models_own_prose_is_used_as_the_report(monkeypatch, no_network):
    """The model writes the report itself, with the pages it read still in context.

    We briefly rebuilt the report from saved notes instead, so that an unsaved fact
    could not appear in it. That is more honest and much thinner — notes are lossy —
    so we went back. The rebuild path survives as the fallback below.
    """
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "save_note",
                        {"claim": "a saved fact", "source_url": "http://a"},
                        "1",
                    )
                ],
            ),
            AIMessage("A detailed report written from everything I read."),
        ]
    )
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm")
    final = app.invoke(graph_mod.initial_state("q"))

    assert final["report"] == "A detailed report written from everything I read."


def test_budget_exhaustion_still_produces_a_report(monkeypatch, no_network):
    """A model that never stops calling tools must still yield a report from notes."""
    never_stops = AIMessage(
        "",
        tool_calls=[
            _call("save_note", {"claim": "a fact", "source_url": "http://example.com/a"}, "1")
        ],
    )
    model = FakeModel([never_stops, never_stops, AIMessage("Forced report from notes.")])
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm", max_steps=2)
    final = app.invoke(graph_mod.initial_state("q"))

    assert final["steps"] == 2  # the budget was actually enforced
    assert final["report"] == "Forced report from notes."
    # Only one note: turn 1's tool calls ran, then turn 2 hit the budget and its
    # pending tool calls were abandoned in favour of writing the report. The budget
    # counts agent turns, not tool executions.
    assert len(final["notes"]) == 1


def test_answering_with_no_notes_gets_nudged_then_accepted(monkeypatch, no_network):
    """The regression this whole nudge mechanism exists for.

    A model that answers straight from memory has saved nothing, so the report would
    be unciteable. The graph pushes back — twice — and only then gives up and takes
    the answer. Three agent turns for one apparent reply.
    """
    model = FakeModel([AIMessage("I already know this.")])
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm", max_nudges=2)
    final = app.invoke(graph_mod.initial_state("q"))

    assert final["nudges"] == 2
    assert final["steps"] == 3  # 1 original answer + 1 per nudge
    assert final["report"] == "I already know this."
    nudge_msgs = [
        m for m in final["messages"] if "not saved any notes" in str(m.content)
    ]
    assert len(nudge_msgs) == 2


def test_a_saved_note_means_no_nudging(monkeypatch, no_network):
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    _call("save_note", {"claim": "c", "source_url": "http://a"}, "1")
                ],
            ),
            AIMessage("Done, with a citation."),
        ]
    )
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm")
    final = app.invoke(graph_mod.initial_state("q"))

    assert final["nudges"] == 0
    assert final["report"] == "Done, with a citation."
    assert len(final["notes"]) == 1


def test_repeated_fetch_is_skipped_across_turns(monkeypatch, no_network):
    """The dedupe guard in run_tools should survive across graph iterations."""
    fetch = AIMessage(
        "", tool_calls=[_call("fetch_page", {"url": "http://example.com/a"}, "1")]
    )
    model = FakeModel([fetch, fetch, AIMessage("done")])
    _install(monkeypatch, model)

    app = graph_mod.build_graph("glm", max_steps=12)
    final = app.invoke(graph_mod.initial_state("q"))

    # fetched once, skipped the second time
    assert final["visited"] == ["http://example.com/a"]
    skipped = [
        m for m in final["messages"] if "[skipped]" in str(getattr(m, "content", ""))
    ]
    assert len(skipped) == 1
