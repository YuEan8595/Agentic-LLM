"""Build the document search index.

Usage:
    1. Put .pdf / .txt / .md files in backend/documents/
    2. Run:  python ingest.py
    3. Restart the server — the agent can now search those documents.

Re-run this whenever your documents change.
"""

from rag import build_index

if __name__ == "__main__":
    build_index()
