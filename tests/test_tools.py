"""Tool behaviour with the network mocked out.

The point of these tests is the failure paths. A tool that raises inside the graph
turns into an opaque agent stall, so every tool must return a string no matter what
the network does.
"""

import httpx

from explorer import tools
from explorer.tools import MAX_PAGE_CHARS, fetch_page, save_note, web_search


class _Resp:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


PAGE = "<html><body><article>{}</article></body></html>"


# --- web_search ------------------------------------------------------------


def test_web_search_formats_results(monkeypatch):
    class _DDGS:
        def text(self, query, **kwargs):
            return [{"title": "T", "href": "http://a", "body": "snippet"}]

    monkeypatch.setattr(tools, "DDGS", _DDGS)
    out = web_search.invoke({"query": "x"})
    assert "[1] T" in out
    assert "http://a" in out
    assert "snippet" in out


def test_web_search_handles_no_results(monkeypatch):
    class _DDGS:
        def text(self, query, **kwargs):
            return []

    monkeypatch.setattr(tools, "DDGS", _DDGS)
    out = web_search.invoke({"query": "x"})
    assert "No results found" in out
    assert "shorter, simpler keywords" in out  # the model needs to know what to change


def test_web_search_passes_backend_and_recency(monkeypatch):
    """The engine chain and the recency filter must actually reach ddgs."""
    seen = {}

    class _DDGS:
        def text(self, query, **kwargs):
            seen.update(kwargs)
            return []

    monkeypatch.setattr(tools, "DDGS", _DDGS)
    web_search.invoke({"query": "x", "recency": "m"})
    assert seen["timelimit"] == "m"
    assert seen["backend"] == tools.SEARCH_BACKEND


def test_web_search_returns_error_string_not_exception(monkeypatch):
    class _DDGS:
        def text(self, query, **kwargs):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(tools, "DDGS", _DDGS)
    out = web_search.invoke({"query": "x"})
    assert out.startswith("[error]")
    assert "rate limited" in out


# --- fetch_page ------------------------------------------------------------


def test_fetch_page_extracts_text(monkeypatch):
    body = "This is the real article content, long enough to be extracted properly."
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: _Resp(PAGE.format(body)))
    out = fetch_page.invoke({"url": "http://a"})
    assert "real article content" in out


def test_fetch_page_truncates(monkeypatch):
    body = "word " * 5000
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: _Resp(PAGE.format(body)))
    out = fetch_page.invoke({"url": "http://a"})
    assert "[...truncated]" in out
    assert len(out) < MAX_PAGE_CHARS + 100


def test_fetch_page_reports_network_failure(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(tools.httpx, "get", _boom)
    out = fetch_page.invoke({"url": "http://a"})
    assert out.startswith("[error]")
    assert "http://a" in out


def test_fetch_page_reports_empty_extraction(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: _Resp("<html></html>"))
    out = fetch_page.invoke({"url": "http://a"})
    assert out.startswith("[error]")


# --- save_note -------------------------------------------------------------


def test_save_note_echoes_the_claim():
    out = save_note.invoke({"claim": "the sky is blue", "source_url": "http://a"})
    assert "the sky is blue" in out


# --- schemas ---------------------------------------------------------------


def test_every_tool_has_a_description():
    """The docstring is the prompt the model reads. An empty one is a silent bug."""
    for tool in tools.TOOLS:
        assert tool.description and len(tool.description) > 20, tool.name
