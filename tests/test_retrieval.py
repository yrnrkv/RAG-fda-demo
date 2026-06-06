from pathlib import Path

from fda_traceability_rag.models import Chunk
from fda_traceability_rag.retrieval import HybridRetriever
from fda_traceability_rag.storage import ChunkStore


def test_bm25_retrieves_exact_regulatory_acronym(tmp_path: Path):
    chunks = [
        Chunk(
            chunk_id="chunk-cte",
            text="A critical tracking event, or CTE, is an event for which records are required.",
            source="qa_guidance",
            source_url="https://example.test/qa",
            title="Traceability Q&A",
            doc_type="Q&A",
            section="Critical Tracking Events",
            entity_type="general",
        ),
        Chunk(
            chunk_id="chunk-farm",
            text="A farm may qualify for certain exemptions based on sales thresholds.",
            source="qa_guidance",
            source_url="https://example.test/qa",
            title="Traceability Q&A",
            doc_type="Q&A",
            section="Farms",
            entity_type="farm",
        ),
    ]
    ChunkStore(tmp_path / "processed" / "chunks.jsonl").write(chunks)

    retriever = HybridRetriever(tmp_path)
    results = retriever.retrieve("What is a CTE?", top_k=1)

    assert results[0].chunk.chunk_id == "chunk-cte"
    assert results[0].bm25_score > 0
