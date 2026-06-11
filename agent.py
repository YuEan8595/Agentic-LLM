"""Builds the agentic LLM using LangChain's create_agent + Gemini 2.5 Flash.

create_agent runs a ReAct-style loop: the model reasons, optionally calls a
tool, sees the result, and repeats until it has a final answer. The
checkpointer gives each conversation (thread_id) its own memory.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from tools import TOOLS

load_dotenv()

# Caps how many graph steps the agent may take in one turn (each tool round is
# ~2 steps: model + tools). Stops a confused model from looping forever and
# burning through the free Gemini quota. Default LangGraph value is 25.
RECURSION_LIMIT = 15

SYSTEM_PROMPT = """You are a helpful, friendly agentic assistant.

You have access to tools. Use them when they genuinely help:
- `calculator` for any arithmetic — never do math in your head.
- `get_current_datetime` when the user asks about the current time or date.
- `search_documents` to answer from the user's OWN files (their notes, PDFs,
  policies, reports). Prefer this whenever the question is about their specific
  or private content.
- `search_wikipedia` for general public facts about people, places, events, or
  concepts that wouldn't be in the user's own documents.

Reason step by step. When a tool would help, call it. After you have what you
need, reply in clear, plain language. If a question needs no tool, just answer.
When you answer from the user's documents, briefly mention the source file.
"""


def build_agent():
    """Construct and return the compiled agent graph."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and paste a key from https://aistudio.google.com/apikey"
        )

    # gemini-2.5-flash has a generous free tier on the Gemini Developer API.
    # max_retries is bounded so a quota error surfaces quickly as a friendly
    # notice instead of backing off silently for a minute or more.
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_retries=2,
    )

    # InMemorySaver keeps conversation memory in RAM (resets on restart).
    # For production, swap in a Postgres/Redis checkpointer.
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
