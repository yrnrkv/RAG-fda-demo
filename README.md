# FDA Traceability RAG

A local-first RAG pipeline for the FDA Food Traceability Rule guidance documents:

- Questions and Answers About the Requirements for Additional Traceability Records for Certain Foods
- Small Entity Compliance Guide: Requirements for Additional Traceability Records for Certain Foods

The project follows a hybrid retrieval design: Q&A-aware chunking, paragraph-window chunking for narrative guidance, Chroma dense vectors, BM25 keyword matching for regulatory acronyms and section numbers, metadata boosts, optional reranking, and citation-aware answer generation.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Build the local index:

```bash
fda-rag ingest
```

Ask a question:

```bash
fda-rag ask "What is a CTE?"
```

Without an API key, `ask` returns an extractive answer from retrieved passages. To enable a synthesized answer:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
fda-rag ask "Is a farm with $20k in tomato sales exempt?"
```

## Commands

```bash
fda-rag ingest              # fetch FDA sources, chunk, embed, and index
fda-rag ingest --from-cache # rebuild from data/raw/*.json
fda-rag search "TLC"        # inspect retrieved chunks only
fda-rag ask "What is a KDE?"# retrieve + generate an answer with citations
fda-rag eval                # small retrieval smoke test
fda-rag status              # show local index status
```

## Pipeline Design

Ingestion downloads the FDA landing pages and prefers linked PDF downloads when available. PDFs are extracted with `pypdf`; HTML fallback extraction uses `BeautifulSoup`.

Chunking is document-specific:

- Q&A guidance is split into question-answer pairs whenever possible.
- The compliance guide is split into paragraph windows with token overlap.
- Each chunk gets metadata for source, section, entity type, document type, chunk id, and original URL.

Retrieval combines:

- Chroma dense search with `BAAI/bge-base-en-v1.5`.
- BM25 sparse search for exact terms such as `CTE`, `KDE`, `TLC`, and `§1.1305`.
- Section and entity boosts inferred from the user query.
- Optional cross-encoder reranking via `--rerank`.

Generation uses `OPENAI_API_KEY` when present. The prompt requires answers to stay inside retrieved FDA context and cite chunk ids and sections. When no key is set, the CLI returns the top retrieved passages as an extractive answer.

## Project Layout

```text
src/fda_traceability_rag/
  sources.py      # FDA fetch and PDF/HTML extraction
  chunking.py     # Q&A and narrative chunking with metadata
  storage.py      # JSONL chunk store and Chroma vector index
  retrieval.py    # hybrid dense + BM25 retrieval
  generation.py   # optional LLM synthesis with citation guardrails
  cli.py          # Typer CLI
tests/
  test_chunking.py
  test_retrieval.py
```

## Evaluation Ideas

Start with:

```bash
fda-rag eval
```

Then manually inspect faithfulness and retrieval recall for targeted questions:

- "Is a farm with $20k in tomato sales exempt?"
- "What records does a small retailer need for shellfish?"
- "What is a CTE?"
- "Does the rule apply to foreign firms?"
- "What are FDA rules on drug labeling?"

For a production-grade evaluation suite, add labeled expected chunks and run faithfulness, answer relevance, and context recall with a tool such as RAGAS.
