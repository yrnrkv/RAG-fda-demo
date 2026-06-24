from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from fda_traceability_rag.models import Chunk


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


class ChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, chunks: Iterable[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps({"text": chunk.text, "metadata": chunk.to_metadata()}) + "\n")

    def read(self) -> list[Chunk]:
        if not self.path.exists():
            return []
        chunks = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                chunks.append(Chunk.from_record(record["text"], record["metadata"]))
        return chunks


class VectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "fda_traceability",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._model: Optional[SentenceTransformer] = None
        self.client = chromadb.PersistentClient(path=str(persist_dir))

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def rebuild(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = self.client.create_collection(self.collection_name, metadata={"hnsw:space": "cosine"})

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embed([chunk.text for chunk in batch])
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.to_metadata() for chunk in batch],
                embeddings=embeddings.tolist(),
            )

        # Write index version marker after successful rebuild
        write_index_version(self.persist_dir.parent)

    def query(self, query: str, n_results: int = 8, where: Optional[dict] = None) -> list[tuple[Chunk, float]]:
        collection = self.client.get_collection(self.collection_name)
        embedding = self.embed([query])[0].tolist()
        result = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        matches = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunk = Chunk.from_record(document, metadata)
            score = 1.0 - float(distance)
            matches.append((chunk, score))
        return matches

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)


def _compute_source_hash(source_path: Path) -> str:
    """Compute SHA256 hash of the source file used to build the index by streaming."""
    if source_path.exists():
        h = hashlib.sha256()
        with source_path.open("rb") as f:
            while True:
                block = f.read(65536)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()[:16]
    return "unknown"


def _hash_and_count(source_path: Path) -> tuple[str, int]:
    """Compute hash and line count in a single pass over the file."""
    h = hashlib.sha256()
    count = 0
    if source_path.exists():
        with source_path.open("rb") as f:
            while True:
                block = f.read(65536)
                if not block:
                    break
                h.update(block)
                count += block.count(b"\n")
    return h.hexdigest()[:16] if source_path.exists() else "unknown", count


def write_index_version(data_dir: Path) -> None:
    """Write a VERSION marker file after a successful index build."""
    chunks_path = data_dir / "processed" / "chunks.jsonl"
    chunks_hash, line_count = _hash_and_count(chunks_path)
    version_info = {
        "version": "1",
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "chunks_hash": chunks_hash,
        "chunk_count": line_count,
    }
    version_path = data_dir / "chroma" / "VERSION"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(json.dumps(version_info, indent=2))


def read_index_version(data_dir: Path) -> dict | None:
    """Read the VERSION marker file. Returns None if index not built."""
    version_path = data_dir / "chroma" / "VERSION"
    if not version_path.exists():
        return None
    return json.loads(version_path.read_text())
