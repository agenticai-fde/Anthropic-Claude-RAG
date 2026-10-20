# PDF RAG with ChromaDB and Claude

Ask questions about a PDF and get answers grounded in its content, with page citations.

- **Vector store:** [ChromaDB](https://www.trychroma.com/) (persistent, local)
- **Embeddings:** Chroma's default `all-MiniLM-L6-v2` (runs locally, no API key needed)
- **LLM:** Anthropic Claude (`claude-haiku-5-5`)

## How it works

1. **Load:** `pypdf` extracts text from each page of the PDF.
2. **Chunk:** text is split into 800-character chunks with 150-character overlap.
3. **Embed and store:** chunks are embedded and saved to `chroma_db/` with their page numbers.
4. **Retrieve:** your question is embedded and the 4 most similar chunks are fetched.
5. **Generate:** Claude answers using only those chunks and cites the pages. If the answer isn't in the PDF, it says so.

## Project structure

```
RAG/
├── rag.py            # indexing, retrieval and Q&A
├── data/
│   └── sample.pdf    # sample document (fictional company handbook)
├── requirements.txt
├── .env.example      # template for your API key
└── .gitignore        # excludes .env and chroma_db/
```

## Setup

1. Install dependencies (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file from the template and add your [Anthropic API key](https://console.anthropic.com/):

   ```bash
   cp .env.example .env
   ```

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

## Usage

```bash
# Interactive Q&A over data/sample.pdf
python rag.py

# Ask a single question
python rag.py -q "How many leave days can I carry forward?"

# Use your own PDF (rebuild the index when switching PDFs)
python rag.py --pdf path/to/your.pdf --reindex
```

On first run the PDF is indexed into `chroma_db/`, and Chroma downloads its embedding model (about 80 MB). Later runs reuse the stored index.

### Example

```
> How many leave days can I carry forward, and what is the learning budget?

You can carry forward up to 8 unused annual leave days to the next year.
The rest lapse on 31 March [1, page 2].

Your annual learning budget is INR 60,000 for courses, certifications and
conferences [2, page 5].

Retrieved: p2, p5, p3, p4
```

## Configuration

Edit the constants at the top of `rag.py`:

| Setting         | Default            | Description                          |
|-----------------|--------------------|--------------------------------------|
| `MODEL`         | `claude-haiku-5-5` | Claude model used for answers        |
| `CHUNK_SIZE`    | `800`              | Characters per chunk                 |
| `CHUNK_OVERLAP` | `150`              | Overlap between consecutive chunks   |
| `TOP_K`         | `4`                | Number of chunks retrieved per query |

## Notes

- Scanned (image-only) PDFs have no extractable text and need OCR first.
- Never commit `.env`. It is already listed in `.gitignore`.
