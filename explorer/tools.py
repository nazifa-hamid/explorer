"""The agent's three tools.

Each docstring IS the tool description the model sees — LangChain derives the schema
from the signature and the description from the docstring. So these are prompt text,
not documentation, and wording them badly shows up as bad tool use.

The tools stay pure: they take arguments and return a string. Anything that mutates
graph state (recording a note, marking a URL visited) happens in the tool *node*, in
graph.py. Keeping the effect at the node boundary makes both sides easy to test.
"""

import os

import httpx
import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

MAX_PAGE_CHARS = 4000
FETCH_TIMEOUT = 20
USER_AGENT = "explorer-research-agent/0.1"

# ddgs fronts several engines. "auto" rotates, which in practice meant a lot of
# DuckDuckGo — fine for general queries, weak on recent technical material. A
# comma-delimited list is a fallback chain: try Google first, fall back when an
# engine rate-limits or returns nothing. These are scrapers, not official APIs, so
# any single one can fail on any given day; the chain is what makes it reliable.
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "google, brave, duckduckgo")


@tool
def web_search(query: str, k: int = 5, recency: str | None = None) -> str:
    """Search the web. Returns numbered results, each with a title, a URL, and a short
    snippet.

    Args:
        query: Plain keywords. Avoid long quoted phrases — they return nothing.
        k: How many results to return.
        recency: Restrict to recent results. One of 'd' (day), 'w' (week),
            'm' (month), 'y' (year). Use this when the question is about current or
            recent events, otherwise leave it unset.

    This returns snippets only, never full pages. Follow up with fetch_page on the
    URLs worth reading.
    """
    try:
        results = list(
            DDGS().text(
                query,
                max_results=k,
                backend=SEARCH_BACKEND,
                timelimit=recency,
            )
        )
    except Exception as e:
        return f"[error] search failed: {e}"
    if not results:
        return (
            "No results found. Try shorter, simpler keywords — long quoted phrases "
            "rarely match anything."
        )
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )


@tool
def fetch_page(url: str) -> str:
    """Fetch a URL and return its main text content with navigation and boilerplate
    stripped. Content is truncated, so prefer fetching a few relevant pages over many
    marginal ones."""
    try:
        resp = httpx.get(
            url,
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"[error] could not fetch {url}: {e}"

    text = trafilatura.extract(resp.text) or ""
    if not text.strip():
        return f"[error] no extractable text at {url}"
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n\n[...truncated]"
    return text


@tool
def set_plan(sub_questions: list[str]) -> str:
    """Break the research question into 2–4 sub-questions you must answer.

    Call this first, before searching. Each sub-question should be one thing you could
    look up and answer on its own — together they should fully cover the question you
    were asked. You are not done until every one of them has at least one note against
    it, so keep the list short and make each entry genuinely necessary.

    You may call this again to revise the plan if research shows you framed it wrong.
    """
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1))
    return f"Plan set. You must gather notes for each of these:\n{numbered}"


@tool
def save_note(claim: str, source_url: str, answers: int | None = None) -> str:
    """Record one factual claim and the URL it came from.

    This is not optional bookkeeping: the final report is assembled from notes, and a
    fact that was never saved cannot appear in it. Call this for every fact you intend
    to use, immediately after reading the page it came from. Keep each claim to a
    single sentence.

    Args:
        claim: One sentence stating the fact.
        source_url: The URL you read it on.
        answers: The number of the sub-question from your plan that this fact helps
            answer. This is how the plan gets marked off, so set it whenever the fact
            belongs to one.
    """
    return f"Saved note: {claim}"


TOOLS = [set_plan, web_search, fetch_page, save_note]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
