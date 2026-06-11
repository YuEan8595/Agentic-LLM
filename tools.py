"""Tools the agent can call.

Each tool is a plain Python function decorated with @tool. The docstring is
important: the model reads it to decide *when* to call the tool, so keep
descriptions clear and specific. All tools here are free and need no API key.

Tools are written so they NEVER raise: on failure they return a short error
string. That way a flaky network call degrades into a message the model can
handle, instead of crashing the whole agent run.
"""

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from langchain_core.tools import tool


# --------------------------------------------------------------------------- #
# Calculator — evaluates arithmetic safely (no eval(), no arbitrary code)
# --------------------------------------------------------------------------- #
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Supports + - * / // % ** and parentheses. Use this for any math instead of
    calculating in your head. Example input: "(3 + 4) * 2 ** 3".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate '{expression}': {exc}"


# --------------------------------------------------------------------------- #
# Current date / time — timezone aware
# --------------------------------------------------------------------------- #
@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Return the current date and time.

    Optionally pass an IANA timezone name such as 'Asia/Kuala_Lumpur',
    'America/New_York', or 'Europe/London'. Defaults to UTC.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:  # noqa: BLE001
        return (
            f"Unknown timezone '{timezone}'. Use an IANA name like "
            "'Asia/Kuala_Lumpur' or 'America/New_York'."
        )
    return datetime.now(tz).strftime("%A, %d %B %Y, %H:%M:%S %Z")


# --------------------------------------------------------------------------- #
# Wikipedia lookup — calls the API directly with a proper User-Agent.
#
# The old `wikipedia` PyPI package is unmaintained and gets empty/blocked
# responses from Wikimedia (which now requires a descriptive User-Agent),
# leading to "Expecting value: line 1 column 1 (char 0)". This version sends a
# real User-Agent and handles every failure gracefully.
# --------------------------------------------------------------------------- #
_WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks for a descriptive UA identifying the app. Put your own contact in.
_WIKI_HEADERS = {
    "User-Agent": "agentic-gemini-demo/1.0 (LangChain tutorial; https://example.com)"
}


@tool
def search_wikipedia(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary.

    Use this for factual questions about people, places, events, or concepts.
    Example input: "Ada Lovelace".
    """
    try:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 1,
        }
        resp = requests.get(_WIKI_API, params=params, headers=_WIKI_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return f"No Wikipedia article found for '{query}'."

        page = next(iter(pages.values()))
        title = page.get("title", query)
        extract = (page.get("extract") or "").strip()
        if not extract:
            return f"Found the article '{title}' but it had no summary text."
        return f"{title}: {extract[:1500]}"
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia lookup failed for '{query}': {exc}"


from rag import search_documents

# The list the agent is built with. Add your own tools here.
TOOLS = [calculator, get_current_datetime, search_wikipedia, search_documents]
