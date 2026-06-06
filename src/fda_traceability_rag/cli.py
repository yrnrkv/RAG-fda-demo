from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from fda_traceability_rag.chunking import chunk_documents
from fda_traceability_rag.generation import AnswerGenerator
from fda_traceability_rag.models import SourceDocument
from fda_traceability_rag.retrieval import HybridRetriever
from fda_traceability_rag.sources import fetch_default_sources
from fda_traceability_rag.storage import ChunkStore, DEFAULT_EMBEDDING_MODEL, VectorStore


app = typer.Typer(help="FDA Food Traceability Rule RAG pipeline.")
console = Console()


@app.command()
def ingest(
    data_dir: Path = typer.Option(Path("data"), help="Directory for raw data and indexes."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformers model."),
    from_cache: bool = typer.Option(False, help="Reuse previously fetched raw source JSON files."),
) -> None:
    """Fetch FDA guidance, chunk it, and build the dense + sparse indexes."""
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if from_cache:
        documents = _load_raw_documents(raw_dir)
        if not documents:
            raise typer.BadParameter("No cached source documents found. Run without --from-cache first.")
    else:
        console.print("[bold]Fetching FDA source documents...[/bold]")
        documents = fetch_default_sources()
        _write_raw_documents(raw_dir, documents)

    console.print("[bold]Chunking documents...[/bold]")
    chunks = chunk_documents(documents)
    ChunkStore(processed_dir / "chunks.jsonl").write(chunks)

    console.print(f"[bold]Building Chroma vector index with {embedding_model}...[/bold]")
    VectorStore(data_dir / "chroma", embedding_model=embedding_model).rebuild(chunks)

    table = Table(title="Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Documents", str(len(documents)))
    table.add_row("Chunks", str(len(chunks)))
    table.add_row("Chunk JSONL", str(processed_dir / "chunks.jsonl"))
    table.add_row("Chroma index", str(data_dir / "chroma"))
    console.print(table)


@app.command()
def search(
    question: str = typer.Argument(..., help="Question to retrieve context for."),
    data_dir: Path = typer.Option(Path("data"), help="Directory containing indexes."),
    top_k: int = typer.Option(6, help="Number of chunks to return."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformers model."),
    rerank: bool = typer.Option(False, help="Use cross-encoder reranking if available."),
) -> None:
    """Show retrieved chunks without calling an LLM."""
    retriever = HybridRetriever(data_dir, embedding_model=embedding_model, use_reranker=rerank)
    results = retriever.retrieve(question, top_k=top_k)
    _print_results(results)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    data_dir: Path = typer.Option(Path("data"), help="Directory containing indexes."),
    top_k: int = typer.Option(6, help="Number of chunks to send to answer generation."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformers model."),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI chat model when OPENAI_API_KEY is set."),
    rerank: bool = typer.Option(False, help="Use cross-encoder reranking if available."),
) -> None:
    """Answer a question with citations from the FDA guidance sources."""
    retriever = HybridRetriever(data_dir, embedding_model=embedding_model, use_reranker=rerank)
    results = retriever.retrieve(question, top_k=top_k)
    answer = AnswerGenerator(model=model).answer(question, results)

    console.print("\n[bold]Answer[/bold]")
    console.print(answer.text)
    console.print(f"\n[dim]Generation mode: {'LLM' if answer.used_llm else 'extractive'}[/dim]")
    _print_citations(answer.citations)


@app.command("eval")
def evaluate(
    data_dir: Path = typer.Option(Path("data"), help="Directory containing indexes."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformers model."),
    top_k: int = typer.Option(5, help="Number of chunks to inspect per question."),
) -> None:
    """Run a small retrieval smoke test over representative regulatory questions."""
    questions = [
        "Is a farm with $20k in tomato sales exempt?",
        "What records does a small retailer need for shellfish?",
        "What is a CTE?",
        "Does the rule apply to foreign firms?",
        "What are FDA rules on drug labeling?",
    ]
    retriever = HybridRetriever(data_dir, embedding_model=embedding_model)
    table = Table(title="Retrieval Evaluation")
    table.add_column("Question")
    table.add_column("Top Section")
    table.add_column("Top Score")
    table.add_column("Top Chunk")
    for question in questions:
        results = retriever.retrieve(question, top_k=top_k)
        top = results[0] if results else None
        table.add_row(
            question,
            top.chunk.section if top else "none",
            f"{top.score:.3f}" if top else "0.000",
            top.chunk.chunk_id if top else "none",
        )
    console.print(table)


@app.command()
def status(data_dir: Path = typer.Option(Path("data"), help="Directory containing indexes.")) -> None:
    """Show local index status."""
    chunks = ChunkStore(data_dir / "processed" / "chunks.jsonl").read()
    table = Table(title="RAG Project Status")
    table.add_column("Item")
    table.add_column("Status")
    table.add_row("Chunks", str(len(chunks)))
    table.add_row("Chunk file", _exists(data_dir / "processed" / "chunks.jsonl"))
    table.add_row("Chroma dir", _exists(data_dir / "chroma"))
    table.add_row("Raw source dir", _exists(data_dir / "raw"))
    console.print(table)


def _write_raw_documents(raw_dir: Path, documents: list[SourceDocument]) -> None:
    for document in documents:
        payload = {
            "source_id": document.source_id,
            "title": document.title,
            "url": document.url,
            "doc_type": document.doc_type,
            "text": document.text,
            "downloaded_from": document.downloaded_from,
        }
        (raw_dir / f"{document.source_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def _load_raw_documents(raw_dir: Path) -> list[SourceDocument]:
    documents = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents.append(SourceDocument(**payload))
    return documents


def _print_results(results) -> None:
    table = Table(title="Retrieved Chunks")
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Section")
    table.add_column("Entity")
    table.add_column("Chunk")
    for index, result in enumerate(results, start=1):
        table.add_row(
            str(index),
            f"{result.score:.3f}",
            result.chunk.section,
            result.chunk.entity_type,
            result.chunk.chunk_id,
        )
    console.print(table)


def _print_citations(citations: list[str]) -> None:
    if not citations:
        return
    console.print("\n[bold]Citations[/bold]")
    for citation in citations:
        console.print(f"- {citation}")


def _exists(path: Path) -> str:
    return "present" if path.exists() else "missing"


if __name__ == "__main__":
    app()
