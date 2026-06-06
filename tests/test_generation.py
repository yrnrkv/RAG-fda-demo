from fda_traceability_rag.generation import AnswerGenerator
from fda_traceability_rag.models import Chunk
from fda_traceability_rag.retrieval import RetrievalResult


def test_answer_falls_back_for_out_of_scope_question(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunk = Chunk(
        chunk_id="chunk-records",
        text="The Food Traceability Rule requires certain records for covered foods.",
        source="small_entity_guide",
        source_url="https://example.test/guide",
        title="Small Entity Guide",
        doc_type="compliance_guide",
        section="Recordkeeping",
    )

    answer = AnswerGenerator().answer(
        "What are FDA rules on drug labeling?",
        [RetrievalResult(chunk=chunk, score=0.95)],
    )

    assert not answer.used_llm
    assert "outside the covered FDA guidance documents" in answer.text


def test_answer_allows_traceability_question(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunk = Chunk(
        chunk_id="chunk-cte",
        text="A CTE is a critical tracking event for which traceability records are required.",
        source="small_entity_guide",
        source_url="https://example.test/guide",
        title="Small Entity Guide",
        doc_type="compliance_guide",
        section="Critical Tracking Events",
    )

    answer = AnswerGenerator().answer(
        "What is a CTE?",
        [RetrievalResult(chunk=chunk, score=0.95)],
    )

    assert "most relevant FDA guidance passages" in answer.text
