from fda_traceability_rag.chunking import chunk_narrative_document, chunk_qa_document
from fda_traceability_rag.models import SourceDocument


def test_qa_chunking_preserves_question_answer_metadata():
    document = SourceDocument(
        source_id="qa_guidance",
        title="Traceability Q&A",
        url="https://example.test/qa",
        doc_type="Q&A",
        text="""
        Farms

        1. Is a farm with $20k in tomato sales exempt?
        A farm may qualify for an exemption depending on the applicable sales thresholds
        and the foods sold during the relevant time period.

        Retail Food Establishments

        2. What records must a retailer keep?
        Retail food establishments must maintain traceability records when covered by the rule.
        """,
    )

    chunks = chunk_qa_document(document)

    assert len(chunks) == 2
    assert chunks[0].question == "Is a farm with $20k in tomato sales exempt?"
    assert chunks[0].section == "Farms"
    assert chunks[0].entity_type == "farm"
    assert "Question:" in chunks[0].text
    assert "Answer:" in chunks[0].text
    assert chunks[1].section == "Retail Food Establishments"


def test_narrative_chunking_adds_overlap_and_entity_metadata():
    paragraph = (
        "Small retail food establishments should understand when records are required "
        "for foods on the Food Traceability List and how traceability lot codes are used. "
    )
    document = SourceDocument(
        source_id="small_entity_guide",
        title="Small Entity Guide",
        url="https://example.test/guide",
        doc_type="compliance_guide",
        text="\n\n".join([paragraph] * 12),
    )

    chunks = chunk_narrative_document(document, max_tokens=60, overlap_tokens=10)

    assert len(chunks) > 1
    assert all(chunk.doc_type == "compliance_guide" for chunk in chunks)
    assert any("retailer" in chunk.entity_type for chunk in chunks)
