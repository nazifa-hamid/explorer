"""Graph state.

The reducers are the whole lesson here: a node returns a *partial* update and
LangGraph merges it according to the annotation. `add_messages` appends instead of
overwriting, so you never hand-manage conversation history.

`notes` is deliberately separate from `messages`. Fetched pages are large and
transient; notes are small, durable, and citable. That split is what lets the agent
research for many steps without its context filling up with raw HTML text, and it is
what `finish` falls back on when the step budget runs out.

`candidates` and `visited` together answer "is there anything left to read?" — the
question the agent could not answer about itself, and the reason its first real run
searched eight times while reading twice.

`plan` answers a different one: "what am I trying to find out, and which parts do I
still lack?" A question string in `messages` records the words; a list of
sub-questions records the *intent*, in a form the graph can check progress against.
Each note may tag the sub-question it answers, which turns the plan into a coverage
checklist for free.

The pattern behind all of them: anything the agent needs to stay oriented has to live
in state. The transcript technically contains it, but the model will not reliably
re-derive its own position from a growing pile of messages.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Note(TypedDict):
    claim: str
    source_url: str
    answers: int | None   # which sub-question (1-based) this fact helps answer


class ResearchState(TypedDict):
    question: str                                    # the original ask, never mutated
    plan: list[str]                                  # sub-questions, replaced wholesale
    messages: Annotated[list[AnyMessage], add_messages]  # appended
    notes: Annotated[list[Note], operator.add]           # appended
    visited: Annotated[list[str], operator.add]          # appended
    candidates: Annotated[list[str], operator.add]       # URLs search surfaced
    steps: int                                       # overwritten each turn
    nudges: int                                      # times the graph pushed back
    report: str | None                               # overwritten by finish
