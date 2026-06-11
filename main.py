"""FastAPI backend.

Endpoints:
  GET  /                  -> serves the chat frontend
  POST /api/chat/stream   -> streams tool steps AND the answer token-by-token (SSE)

We stream with stream_mode=["updates", "messages"]:
  - "updates"  gives us tool calls + tool results (one event per node step)
  - "messages" gives us the final answer token-by-token as Gemini generates it
A final "answer" event sends the authoritative full text to correct any
streaming artifacts.
"""

import json
import os
import re
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from agent import RECURSION_LIMIT, build_agent

app = FastAPI(title="Agentic LLM — Gemini 2.5 Flash + LangChain")

# CORS is open here for convenience. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()

FRONTEND = os.path.join(os.path.dirname(__file__), "static", "index.html")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _extract_text(content) -> str:
    """Gemini content can be a plain string OR a list of parts like
    [{'type': 'text', 'text': '...'}]. Normalize both to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return ""


def _parse_doc_sources(text: str) -> list[str]:
    """Pull the source filenames out of a search_documents result.

    The tool formats each hit as "[n] filename (p.X)\\n<excerpt>", so we read
    the first line of each block and dedupe."""
    sources: list[str] = []
    for block in text.split("\n\n"):
        first = block.strip().splitlines()[0] if block.strip() else ""
        match = re.match(r"^\[\d+\]\s+(.+)$", first)
        if match:
            name = match.group(1).strip()
            if name not in sources:
                sources.append(name)
    return sources


def _is_rate_limit(exc: Exception) -> bool:
    """Detect a Gemini quota / rate-limit error across SDK versions by inspecting
    the message (the exact exception class varies)."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "resource_exhausted", "quota", "rate limit", "ratelimit")
    )


def _event_stream(message: str, session_id: str):
    """Run the agent and yield SSE events: tool steps, live tokens, final answer."""
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": session_id},
    }
    inputs = {"messages": [{"role": "user", "content": message}]}
    final_text = ""

    try:
        for mode, chunk in agent.stream(
            inputs, config=config, stream_mode=["updates", "messages"]
        ):
            # ---- tool calls + tool results -------------------------------- #
            if mode == "updates":
                for node_output in chunk.values():
                    if not isinstance(node_output, dict):
                        continue
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage):
                            for call in (msg.tool_calls or []):
                                yield _sse("tool_call", {
                                    "name": call.get("name"),
                                    "args": call.get("args", {}),
                                })
                            # The final answer: text, no tool calls.
                            text = _extract_text(msg.content)
                            if text and not msg.tool_calls:
                                final_text = text
                        elif isinstance(msg, ToolMessage):
                            result = str(msg.content)
                            yield _sse("tool_result", {
                                "name": msg.name,
                                "result": result,
                            })
                            # Surface document citations to the UI.
                            if msg.name == "search_documents":
                                sources = _parse_doc_sources(result)
                                if sources:
                                    yield _sse("sources", {"names": sources})

            # ---- live answer tokens --------------------------------------- #
            elif mode == "messages":
                message_chunk, metadata = chunk
                # Tools don't emit LLM tokens; skip just in case.
                if metadata.get("langgraph_node") == "tools":
                    continue
                token = _extract_text(getattr(message_chunk, "content", ""))
                if token:
                    yield _sse("token", {"content": token})

        # Authoritative final text (fills the bubble if no tokens streamed,
        # and corrects any duplicate-token artifacts).
        yield _sse("answer", {"content": final_text})
    except GraphRecursionError:
        yield _sse("error", {
            "kind": "notice",
            "message": (
                "I got stuck taking too many steps on that one and stopped to "
                "avoid looping. Try rephrasing it or breaking it into smaller "
                "questions."
            ),
        })
    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit(exc):
            yield _sse("error", {
                "kind": "notice",
                "message": (
                    "I'm being rate-limited by the Gemini free tier right now. "
                    "Give it a minute and try again."
                ),
            })
        else:
            yield _sse("error", {"kind": "error", "message": str(exc)})
    finally:
        yield _sse("done", {})


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _event_stream(req.message, session_id),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id, "Cache-Control": "no-cache"},
    )


@app.get("/api/history/{session_id}")
def history(session_id: str):
    """Return the visible conversation (user + final assistant turns) that the
    SQLite checkpointer has persisted for this session, so the UI can restore it
    after a reload or server restart."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = agent.get_state(config)
    except Exception:  # noqa: BLE001
        return {"messages": []}

    values = getattr(state, "values", None) or {}
    out = []
    for msg in values.get("messages", []):
        if isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": _extract_text(msg.content)})
        elif isinstance(msg, AIMessage):
            text = _extract_text(msg.content)
            if text and not msg.tool_calls:  # skip tool-call-only messages
                out.append({"role": "assistant", "content": text})
    return {"messages": out}


@app.get("/")
def index():
    return FileResponse(FRONTEND)
