from __future__ import annotations

import hashlib
import re
from typing import Iterable

from fda_traceability_rag.models import Chunk, SourceDocument


SECTION_KEYWORDS = {
    "farms": "Farms",
    "farm": "Farms",
    "fishing vessel": "Fishing Vessels",
    "fishing vessels": "Fishing Vessels",
    "shellfish": "Shellfish",
    "retail": "Retail Food Establishments",
    "restaurant": "Restaurants",
    "restaurants": "Restaurants",
    "recordkeeping": "Recordkeeping",
    "records": "Recordkeeping",
    "transformation": "Critical Tracking Events",
    "critical tracking event": "Critical Tracking Events",
    "cte": "Critical Tracking Events",
    "key data element": "Key Data Elements",
    "kde": "Key Data Elements",
    "traceability lot code": "Traceability Lot Codes",
    "tlc": "Traceability Lot Codes",
    "exemption": "Exemptions",
    "exemptions": "Exemptions",
    "foreign": "Foreign Entities",
    "imports": "Foreign Entities",
}

ENTITY_KEYWORDS = {
    "farm": "farm",
    "farms": "farm",
    "retail food establishment": "retailer",
    "retailer": "retailer",
    "retailers": "retailer",
    "restaurant": "restaurant",
    "restaurants": "restaurant",
    "fishing vessel": "fishing_vessel",
    "fishing vessels": "fishing_vessel",
    "shellfish": "shellfish",
    "foreign": "foreign_entity",
    "importer": "foreign_entity",
    "small entity": "small_entity",
    "small business": "small_entity",
    "warehouse": "warehouse",
    "distributor": "distributor",
    "transporter": "transporter",
}

QUESTION_RE = re.compile(
    r"^(?:(?:Q(?:uestion)?\.?\s*)|(?:[A-Z]?\d{1,3}[.)]\s+)|(?:[A-Z]\.\d+[.)]?\s+))(.+\?)",
    re.IGNORECASE,
)


def chunk_documents(documents: Iterable[SourceDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        if document.doc_type.lower() == "q&a":
            chunks.extend(chunk_qa_document(document))
        else:
            chunks.extend(chunk_narrative_document(document))
    return chunks


def chunk_qa_document(document: SourceDocument) -> list[Chunk]:
    paragraphs = _paragraphs(document.text)
    chunks: list[Chunk] = []
    current_section = "General"
    pending_question: str | None = None
    pending_answer_parts: list[str] = []

    def flush() -> None:
        nonlocal pending_question, pending_answer_parts
        if not pending_question:
            return
        answer = "\n\n".join(part.strip() for part in pending_answer_parts if part.strip())
        if len(answer.split()) < 5:
            return
        text = f"Question: {pending_question}\n\nAnswer: {answer}"
        chunks.append(
            _make_chunk(
                document=document,
                text=text,
                index=len(chunks),
                section=current_section,
                question=pending_question,
                answer=answer,
                strategy="qa_pair",
            )
        )
        pending_question = None
        pending_answer_parts = []

    for paragraph in paragraphs:
        section = infer_section(paragraph)
        if _looks_like_heading(paragraph) and section != "General":
            flush()
            current_section = section
            continue

        question_match = QUESTION_RE.match(paragraph)
        if question_match:
            flush()
            pending_question = _clean_question(question_match.group(1))
            remainder = paragraph[question_match.end(1) :].strip(" -:\n")
            pending_answer_parts = [remainder] if remainder else []
            q_section = infer_section(pending_question)
            if q_section != "General":
                current_section = q_section
            continue

        if pending_question:
            pending_answer_parts.append(paragraph)
        elif len(paragraph.split()) >= 40:
            inferred = infer_section(paragraph)
            chunks.append(
                _make_chunk(
                    document=document,
                    text=paragraph,
                    index=len(chunks),
                    section=inferred if inferred != "General" else current_section,
                    strategy="qa_context",
                )
            )

    flush()
    return chunks


def chunk_narrative_document(
    document: SourceDocument, max_tokens: int = 220, overlap_tokens: int = 50
) -> list[Chunk]:
    paragraphs = _paragraphs(document.text)
    chunks: list[Chunk] = []
    current_section = "General"
    window: list[str] = []
    token_count = 0

    def flush() -> None:
        nonlocal window, token_count
        text = "\n\n".join(window).strip()
        if len(text.split()) < 25:
            return
        chunks.append(
            _make_chunk(
                document=document,
                text=text,
                index=len(chunks),
                section=infer_section(text) if infer_section(text) != "General" else current_section,
                strategy="paragraph_window",
            )
        )
        overlap = _tokens(text)[-overlap_tokens:] if overlap_tokens else []
        window = [" ".join(overlap)] if overlap else []
        token_count = len(overlap)

    for paragraph in paragraphs:
        section = infer_section(paragraph)
        if _looks_like_heading(paragraph) and section != "General":
            flush()
            current_section = section
            continue

        tokens = _tokens(paragraph)
        if token_count + len(tokens) > max_tokens and window:
            flush()
        window.append(paragraph)
        token_count += len(tokens)

    flush()
    return chunks


def infer_section(text: str) -> str:
    lowered = text.lower()
    for keyword, section in SECTION_KEYWORDS.items():
        if keyword in lowered:
            return section
    return "General"


def infer_entity_type(text: str) -> str:
    lowered = text.lower()
    entities = sorted({entity for keyword, entity in ENTITY_KEYWORDS.items() if keyword in lowered})
    return ",".join(entities) if entities else "general"


def _make_chunk(
    document: SourceDocument,
    text: str,
    index: int,
    section: str,
    strategy: str,
    question: str | None = None,
    answer: str | None = None,
) -> Chunk:
    chunk_id = _chunk_id(document.source_id, index, text)
    return Chunk(
        chunk_id=chunk_id,
        text=_clean_text(text),
        source=document.source_id,
        source_url=document.url,
        title=document.title,
        doc_type=document.doc_type,
        section=section,
        entity_type=infer_entity_type(text),
        question=question,
        answer=answer,
        metadata={
            "chunk_index": index,
            "chunking_strategy": strategy,
            "downloaded_from": document.downloaded_from or document.url,
        },
    )


def _chunk_id(source_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}:{index:04d}:{digest}"


def _paragraphs(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.match(r"^\[Page \d+\]$", line):
            lines.append("")
            continue
        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(_clean_text(" ".join(current)))
                current = []
            continue

        starts_new = _looks_like_heading(line) or bool(QUESTION_RE.match(line))
        if starts_new and current:
            paragraphs.append(_clean_text(" ".join(current)))
            current = []
        current.append(line)

    if current:
        paragraphs.append(_clean_text(" ".join(current)))
    return [paragraph for paragraph in paragraphs if paragraph]


def _looks_like_heading(text: str) -> bool:
    words = text.split()
    if len(words) > 12 or len(words) < 1:
        return False
    if text.endswith(".") and not re.match(r"^[IVXLC]+\.", text):
        return False
    lowered = text.lower().strip(":")
    if any(keyword == lowered or keyword in lowered for keyword in SECTION_KEYWORDS):
        return True
    return bool(re.match(r"^(?:[IVXLC]+\.|[A-Z]\.)\s+[A-Z]", text))


def _clean_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w§.-]+\b", text)
