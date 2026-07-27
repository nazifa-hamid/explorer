"""Model registry.

Nothing else in the codebase imports a chat model directly; every caller goes through
`get_model`. That is what will make Phase 2's per-node model assignment a one-liner.

GLM and llama-server both speak the OpenAI protocol, so they share one client class.
Gemini is the exception, and the reason is instructive — see `_gemini`. "OpenAI
compatible" holds right up until a provider needs to round-trip something the
protocol has no field for.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

TEMPERATURE = 0.3

# Generous, because a truncated tool call breaks the loop in a confusing way:
# the model emits half a function call, nothing parses, and the agent looks broken
# when it merely ran out of room.
MAX_TOKENS = 2048

# Free tiers throttle aggressively, and a research run fires many requests back to
# back. The OpenAI client retries 429s on its own, so give it room to do so.
MAX_RETRIES = 6


def _glm():
    """GLM-4.7-Flash via z.ai. Free tier, 128K context, one concurrent request."""
    return ChatOpenAI(
        model=os.getenv("GLM_MODEL", "glm-4.7-flash"),
        base_url="https://api.z.ai/api/paas/v4/",
        api_key=os.environ["ZAI_API_KEY"],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        max_retries=MAX_RETRIES,
        timeout=120,
    )


def _local():
    """Gemma 4 via llama-server.

    Start the server with --jinja, or it will not apply the tool-call chat template
    and every tool call arrives as plain prose in `content` — the loop then never
    fires and the agent just talks to itself.

        llama-server --model <gguf> --host 127.0.0.1 --port 8080 \\
          --n-gpu-layers 99 --ctx-size 32768 --parallel 1 --jinja
    """
    return ChatOpenAI(
        model=os.getenv("LOCAL_MODEL", "gemma-4-e4b"),  # llama-server ignores this
        base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8080/v1"),
        api_key="not-needed",  # the client requires a value; llama-server ignores it
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


def _gemini():
    """Gemini Flash through Google's native API.

    Deliberately NOT the OpenAI-compatible endpoint, even though it exists and would
    keep this file down to one client class. Gemini's thinking models sign their
    reasoning and attach a `thought_signature` to every tool call, in a non-standard
    field (`tool_calls[N].extra_content.google.thought_signature`). Google requires
    that signature be echoed back verbatim on the next request. Standard OpenAI
    clients drop unknown fields, so the signature is lost and turn 2 of any tool loop
    fails with:

        400 INVALID_ARGUMENT — Function call is missing a thought_signature
        in functionCall parts

    A single-turn call succeeds, which makes this look fine until the agent tries to
    continue. The native client round-trips the signature correctly.

    Pin a model rather than using a "-latest" alias: aliases resolve to the newest
    release, and the newest release tends to carry the smallest free-tier daily quota.
    `gemini-flash-latest` resolved to a model capped at 20 requests a day, which a
    single research run exhausts. Quota is per model, so switching gets a fresh one.

    Do not guess model IDs either — they are not derivable ("gemini-3-flash" is a 404,
    "gemini-3.6-flash" is real) and old ones get withdrawn ("gemini-2.5-flash" now
    404s for new keys). `run-explorer models` asks the API what this key can actually
    use. One generation behind the newest is the sweet spot: the newest has the
    smallest free quota, and the oldest may be gone.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        max_retries=MAX_RETRIES,
    )


MODELS = {"glm": _glm, "local": _local, "gemini": _gemini}


def get_model(name: str) -> Any:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    try:
        return MODELS[name]()
    except KeyError as e:  # a missing API key, surfaced as something readable
        raise RuntimeError(
            f"model {name!r} needs the {e.args[0]} environment variable; "
            "copy .env.example to .env and fill it in"
        ) from e
