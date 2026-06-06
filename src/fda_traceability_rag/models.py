from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SourceDocument:
    """A fetched FDA source after text extraction."""

    source_id: str
    title: str
    url: str
    doc_type: str
    text: str
    downloaded_from: Optional[str] = None


@dataclass(frozen=True)
class Chunk:
    """An atomic retrieval unit with citation-friendly metadata."""

    chunk_id: str
    text: str
    source: str
    source_url: str
    title: str
    doc_type: str
    section: str
    entity_type: str = "general"
    question: Optional[str] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("text")
        extra = payload.pop("metadata")
        payload.update(extra)
        return {key: "" if value is None else value for key, value in payload.items()}

    @classmethod
    def from_record(cls, text: str, metadata: Dict[str, Any]) -> "Chunk":
        known = {
            "chunk_id",
            "source",
            "source_url",
            "title",
            "doc_type",
            "section",
            "entity_type",
            "question",
            "answer",
        }
        kwargs = {key: metadata.get(key) or None for key in known}
        kwargs["text"] = text
        kwargs["entity_type"] = kwargs["entity_type"] or "general"
        kwargs["section"] = kwargs["section"] or "Uncategorized"
        kwargs["metadata"] = {key: value for key, value in metadata.items() if key not in known}
        return cls(**kwargs)  # type: ignore[arg-type]
