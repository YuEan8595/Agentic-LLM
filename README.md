# Agentic LLM — Gemini 2.5 Flash + LangChain + FastAPI

A minimal but complete agentic chatbot:

- **Backend:** Python + FastAPI
- **Agent:** LangChain `create_agent` (ReAct loop) on **Google Gemini 2.5 Flash** (free tier)
- **Tools:** calculator, current date/time, Wikipedia lookup — all free, no extra keys
- **Memory:** per-conversation, via a LangGraph checkpointer
- **Frontend:** a single HTML file that streams the agent's tool steps live

```
agentic-gemini/
│─── main.py            FastAPI app + SSE streaming endpoint
│─── agent.py           Gemini model + create_agent setup
│─── tools.py           the tools the agent can call
│─── requirements.txt   
│─── .env
└── static/
    └── index.html         chat UI (served by the backend)
```

## 1. Get a free API key

Go to **https://aistudio.google.com/apikey**, create a key (no billing required for
the free tier), and copy it.

## 2. Install

```bash
cd agentic-gemini
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your key

```bash
cp .env.example .env
# open .env and paste your key into GOOGLE_API_KEY=...
```

## 4. Run

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

## How it works

When you send a message, the backend runs the agent loop. The model decides
whether it needs a tool, calls it, reads the result, and repeats until it has an
answer. Each of those steps is streamed to the browser as a Server-Sent Event,
so you can watch the agent reason and act in real time before the final reply
appears.

Conversation memory is keyed by `session_id`, which the frontend stores in
`localStorage` so a thread continues across reloads. Memory is persisted
server-side by a `SqliteSaver` checkpointer (`backend/checkpoints.db`), so
conversations survive restarts. The "New chat" button starts a fresh thread.

## Search your own documents (RAG)

The agent can answer from your own files using the `search_documents` tool.

1. Put `.pdf`, `.txt`, or `.md` files in `backend/documents/` (a sample
   `sample-handbook.md` is already there).
2. Build the index (uses Google's free `gemini-embedding-001` model):

   ```bash
   cd backend
   python ingest.py
   ```

3. Start the server and ask away. With the sample file indexed, try:
   *"How many vacation days do I get at Northwind, and what are the office
   locations?"* — you'll see the agent call `search_documents` and answer from
   the handbook, citing the source file.

Re-run `python ingest.py` whenever your documents change. The index persists to
`vectorstore/` so it survives restarts. Large document sets can hit the
free embedding rate limits; if that happens, index in smaller batches.

## Extending it

**Add a tool** — write a function in `tools.py`, decorate it with `@tool`, give it
a clear docstring (the model uses it to decide when to call the tool), and add it
to the `TOOLS` list:

```python
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    ...
```

**Scale memory for production** — `agent.py` already persists to SQLite via
`SqliteSaver`. SQLite uses file-level locking, so for multiple server workers or
heavy concurrency, switch to LangGraph's `PostgresSaver`.

**Switch models** — change `model="gemini-2.5-flash"` in `agent.py` to another
Gemini model, or swap `ChatGoogleGenerativeAI` for a different provider's chat
class; the rest of the agent code stays the same.

## Free-tier notes

Gemini 2.5 Flash has a generous free tier on the Gemini Developer API, but it is
rate-limited (requests per minute/day). If you hit a quota error, wait a moment
or check your limits in Google AI Studio.
