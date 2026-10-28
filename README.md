# Advance-RAG

Chat with your own documents. Upload PDF, DOCX, TXT or Markdown files and ask questions about them in a Streamlit chat. Answers come only from your files and cite the passage and page they used.

It goes beyond a basic "embed and search" RAG. It uses parent-child chunking, multi-query rewriting, hybrid search (FAISS + BM25), cross-encoder reranking and an LLM router that decides whether a message needs a search at all.

## Features

- **Upload and manage files** from the sidebar: add several at once, remove one, or clear everything.
- **Parent-child chunking**: small child chunks (400 chars) are embedded and searched for precision. The bigger parent chunk (1600 chars) around each match is what the LLM reads.
- **Multi-query retrieval**: the LLM rewrites the question into a few variants and resolves "it" and "that" from the chat history. Questions in Roman Urdu or other languages are also searched in English.
- **Hybrid search**: FAISS (semantic) and BM25 (keyword) results for every query are merged with Reciprocal Rank Fusion.
- **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` scores each chunk against the question and drops weak matches.
- **Message routing**: each message is classified as one of three types:
  - `SEARCH`: a question about the content
  - `REWRITE`: "make that shorter", "translate it"
  - `CHAT`: greetings, "what files do you have?"
- **File overviews**: a short LLM-written summary of each file, so broad questions like "what is this book about?" get good answers.
- **Citations**: every fact is tagged `[1]`, `[2]`, …, and only the sources the answer actually cites are shown, with PDF page numbers.
- **Conversation memory**: the last few turns are kept for follow-up questions.
- **Streaming answers**, in the same language and script as the question.

## How it works

```
                    ┌──────────────── Ingestion ────────────────┐
 Upload file  ──▶   load text ─▶ parent chunks ─▶ child chunks
                                     │                 │
                                     ▼                 ▼
                              doc_store.json     FAISS (embeddings)
                              (parents, children     + BM25
                               for BM25, overviews)
                    └───────────────────────────────────────────┘

                    ┌──────────────── Question ─────────────────┐
 Question ──▶ Router ──▶ CHAT     ─▶ friendly reply + file list
                    ├──▶ REWRITE  ─▶ reshape the previous answer
                    └──▶ SEARCH   ─▶ multi-query rewrite
                                     ─▶ FAISS + BM25 per query
                                     ─▶ Reciprocal Rank Fusion
                                     ─▶ cross-encoder rerank
                                     ─▶ swap children for parents
                                     ─▶ answer with citations
                    └───────────────────────────────────────────┘
```

## Tech stack

| Part | Tool |
|---|---|
| UI | Streamlit |
| Framework | LangChain |
| Chat model | Groq (default `openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally on CPU) |
| Vector store | FAISS |
| Keyword search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| File loaders | `pypdf`, `docx2txt` |

## Project structure

```
Advance-Rag/
├── app.py               # Streamlit UI: upload, file list, chat
├── chatbot.py           # Router, prompts, context building, citations, streaming
├── retriever.py         # Multi-query, hybrid search, RRF, reranking, parent lookup
├── ingest.py            # Chunking, FAISS index, doc store, file overviews, removal
├── document_loader.py   # PDF / DOCX / TXT / MD text extraction
├── llmClient.py         # Chat model and embedding model setup
├── config.py            # All settings, read from .env with defaults
├── requirements.txt
└── data/
    ├── documents/       # Uploaded files
    └── evaluation/
```

`vectorstore/` is created on first upload and is not committed.

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/AnsRehman7/Advance-RAG.git
cd Advance-RAG

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the project root:

```env
LLM_API_KEY=your_groq_api_key
```

You can get a free key at [console.groq.com](https://console.groq.com/keys). The key is the only required setting. The embedding and reranking models download from Hugging Face on first run and need no key.

### 3. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Upload files in the sidebar, click **Process**, then start asking questions.

## Configuration

Every setting can be overridden in `.env`. If a setting is missing, the default is used.

| Variable | Default | What it does |
|---|---|---|
| `LLM_API_KEY` | (none) | Groq API key (required) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Chat model on Groq |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | `1600` / `200` | Size of the chunks the LLM reads |
| `CHILD_CHUNK_SIZE` / `CHILD_CHUNK_OVERLAP` | `400` / `60` | Size of the chunks that are searched |
| `NUM_QUERY_VARIANTS` | `3` | Question rewrites per search (`0` turns multi-query off) |
| `VECTOR_TOP_K` | `6` | FAISS results per query |
| `BM25_TOP_K` | `6` | BM25 results per query |
| `HYBRID_TOP_K` | `10` | Chunks kept after fusion |
| `RERANKER` | `cross-encoder` | Set to anything else to skip reranking |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `RERANK_TOP_K` | `4` | Parent chunks sent to the LLM |
| `MIN_RERANK_SCORE` | `-4.0` | Chunks scoring below this are dropped |
| `MAX_CONTEXT_CHARS` | `8000` | Maximum characters of passages in the prompt |
| `MEMORY_WINDOW` | `4` | Chat turns remembered (`0` turns memory off) |
| `SUMMARY_INPUT_CHARS` | `8000` | Characters from the start of each file used to write its overview |

If you change the chunk sizes or the embedding model, clear the index so it is rebuilt with the new settings. Use **Remove all documents** in the app, then upload the files again.

## Notes

- Scanned PDFs that contain only images have no extractable text, so those pages are skipped.
- Never commit your `.env` file. It is already listed in `.gitignore`.
