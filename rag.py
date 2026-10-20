"""Simple RAG over a PDF: ChromaDB for vector storage, Anthropic Claude for answers.

Usage:
    python rag.py                      # interactive Q&A over data/sample.pdf
    python rag.py --pdf path\\to.pdf    # use another PDF
    python rag.py --reindex            # rebuild the vector store
    python rag.py -q "How many leave days do I get?"
"""
import argparse
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # ANTHROPIC_API_KEY (see .env.example)

MODEL = "claude-haiku-5-5"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION = "pdf_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def index_pdf(collection, pdf_path: Path) -> None:
    reader = PdfReader(pdf_path)
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        for i, chunk in enumerate(chunk_text(page.extract_text() or "")):
            ids.append(f"{pdf_path.name}-p{page_no}-c{i}")
            docs.append(chunk)
            metas.append({"source": pdf_path.name, "page": page_no})
    if not docs:
        sys.exit(f"No text extracted from {pdf_path} (scanned PDF?)")
    # Chroma's default embedding model (all-MiniLM-L6-v2) is downloaded on first use.
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(docs)} chunks from {pdf_path.name} ({len(reader.pages)} pages)")


def retrieve(collection, question: str, k: int = TOP_K) -> list[tuple[str, dict]]:
    res = collection.query(query_texts=[question], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(client: anthropic.Anthropic, question: str, hits: list[tuple[str, dict]]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {m['source']}, page {m['page']})\n{doc}"
        for i, (doc, m) in enumerate(hits, start=1)
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "Answer the user's question using only the provided context. "
            "Cite sources as [n] with page numbers. If the context does not contain "
            "the answer, say you don't know."
        ),
        messages=[{
            "role": "user",
            "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}",
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over a PDF with ChromaDB + Claude")
    parser.add_argument("--pdf", type=Path, default=HERE / "data" / "sample.pdf")
    parser.add_argument("--reindex", action="store_true", help="rebuild the vector store")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"{args.pdf} not found. Pass a PDF with --pdf")

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if args.reindex:
        try:
            db.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = db.get_or_create_collection(COLLECTION)
    if collection.count() == 0 or args.reindex:
        index_pdf(collection, args.pdf)

    client = anthropic.Anthropic()

    def ask(q: str) -> None:
        hits = retrieve(collection, q)
        print("\n" + answer(client, q, hits))
        print("\nRetrieved:", ", ".join(f"p{m['page']}" for _, m in hits), "\n")

    if args.question:
        ask(args.question)
        return

    print("Ask questions about the PDF (empty line or 'exit' to quit).")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        ask(q)


if __name__ == "__main__":
    main()
