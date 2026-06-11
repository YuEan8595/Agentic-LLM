"""Retrieval-Augmented Generation (RAG).

Lets the agent answer from the user's OWN documents:
  1. Drop .pdf / .txt / .md files into documents/
  2. Run `python ingest.py` to chunk, embed, and store them in a local Chroma index
  3. The agent's `search_documents` tool retrieves relevant passages at query time

Uses Chroma for the vector store. It persists to vectorstore/ so the
index survives restarts. Embeddings use Google's `gemini-embedding-001`
(free tier, same API key as the chat model).

A module-level lock serializes access to the store. The agent can fire several
`search_documents` calls in parallel, and Chroma's native client is not safe
under concurrent access from multiple threads — the lock prevents the
"'RustBindingsAPI' object has no attribute 'bindings'" crash that otherwise
happens when two searches hit the store at the same time.
"""

import os
import shutil
import threading

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_HERE = os.path.dirname(__file__)
DOCS_DIR = os.path.join(_HERE, "documents")
VECTORSTORE_DIR = os.path.join(_HERE, "vectorstore")
COLLECTION = "user_documents"

# Chroma stores its data in a chroma.sqlite3 file inside VECTORSTORE_DIR; its
# presence tells us whether an index has been built yet.
_CHROMA_DB_FILE = os.path.join(VECTORSTORE_DIR, "chroma.sqlite3")

# Serializes all reads/writes against the Chroma client (see module docstring).
_lock = threading.Lock()


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # text-embedding-004 is deprecated; gemini-embedding-001 is the current model.
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


# Cached store handle for the running server (opened once, read many times).
_vectorstore = None


def get_vectorstore():
    """Open the persisted Chroma index, or return None if it hasn't been built."""
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(_CHROMA_DB_FILE):
            return None
        _vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=VECTORSTORE_DIR,
        )
    return _vectorstore


# --------------------------------------------------------------------------- #
# Ingestion (run via ingest.py)
# --------------------------------------------------------------------------- #
def _load_documents(folder: str) -> list[Document]:
    """Read .pdf / .txt / .md files into LangChain Documents."""
    docs: list[Document] = []
    if not os.path.isdir(folder):
        return docs

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        try:
            if ext == "pdf":
                import pypdf

                reader = pypdf.PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if text:
                        docs.append(
                            Document(page_content=text,
                                     metadata={"source": name, "page": i + 1})
                        )
            elif ext in ("txt", "md", "markdown"):
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read().strip()
                if text:
                    docs.append(Document(page_content=text, metadata={"source": name}))
            else:
                print(f"  - skipping unsupported file: {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed to read {name}: {exc}")
    return docs


def build_index() -> None:
    """Rebuild the Chroma index from everything in DOCS_DIR."""
    docs = _load_documents(DOCS_DIR)
    if not docs:
        print(f"No readable documents in {DOCS_DIR}. Add .pdf/.txt/.md files and retry.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} document(s) -> {len(chunks)} chunks. Embedding...")

    # Start clean so re-running doesn't create duplicates.
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"Done. Indexed {len(chunks)} chunks into {VECTORSTORE_DIR}")


# --------------------------------------------------------------------------- #
# The tool the agent calls
# --------------------------------------------------------------------------- #
@tool
def search_documents(query: str) -> str:
    """Search the user's own uploaded documents for relevant passages.

    Use this whenever the question might be answered by the user's own files —
    their notes, PDFs, manuals, policies, or reports. Returns the most relevant
    excerpts together with their source filenames. Prefer this over general
    knowledge when the user asks about their specific or private content.
    """
    try:
        # Hold the lock across open + search so parallel tool calls can't race
        # the native Chroma client.
        with _lock:
            store = get_vectorstore()
            if store is None:
                return (
                    "No document index found yet. Add files to backend/documents/ "
                    "and run `python ingest.py`, then try again."
                )
            results = store.similarity_search(query, k=4)
    except Exception as exc:  # noqa: BLE001
        return f"Document search failed: {exc}"

    if not results:
        return "No relevant passages found in the indexed documents."

    blocks = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        where = source + (f" (p.{page})" if page is not None else "")
        blocks.append(f"[{i}] {where}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)
