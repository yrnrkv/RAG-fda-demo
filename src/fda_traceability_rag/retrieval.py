from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from fda_traceability_rag.chunking import infer_entity_type, infer_section
from fda_traceability_rag.models import Chunk
from fda_traceability_rag.storage import ChunkStore, DEFAULT_EMBEDDING_MODEL, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None


class HybridRetriever:
    def __init__(
        self,
        data_dir: Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.data_dir = data_dir
        self.chunk_store = ChunkStore(data_dir / "processed" / "chunks.jsonl")
        self.vector_store = VectorStore(data_dir / "chroma", embedding_model=embedding_model)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self._chunks: Optional[list[Chunk]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: Optional[list[list[str]]] = None

    @property
    def chunks(self) -> list[Chunk]:
        if self._chunks is None:
            self._chunks = self.chunk_store.read()
        return self._chunks

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._tokenized = [_tokenize(chunk.text) for chunk in self.chunks]
            self._bm25 = BM25Okapi(self._tokenized)
        return self._bm25

    def retrieve(self, query: str, top_k: int = 6, candidate_k: int = 24) -> list[RetrievalResult]:
        if not self.chunks:
            raise RuntimeError("No chunks found. Run `fda-rag ingest` first.")

        dense = self._dense_candidates(query, candidate_k)
        sparse = self._bm25_candidates(query, candidate_k)
        results = self._merge(query, dense, sparse)

        if self.use_reranker and results:
            results = self._rerank(query, results[:candidate_k])

        return results[:top_k]

    def _dense_candidates(self, query: str, candidate_k: int) -> dict[str, tuple[Chunk, float]]:
        where = _section_filter(query)
        try:
            matches = self.vector_store.query(query, n_results=candidate_k, where=where)
            if where and len(matches) < max(3, candidate_k // 4):
                matches = self.vector_store.query(query, n_results=candidate_k)
        except Exception:
            matches = []
        return {chunk.chunk_id: (chunk, max(score, 0.0)) for chunk, score in matches}

    def _bm25_candidates(self, query: str, candidate_k: int) -> dict[str, tuple[Chunk, float]]:
        tokens = _tokenize(query)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        if not len(scores):
            return {}
        query_terms = set(tokens)
        adjusted_scores = []
        for index, score in enumerate(scores):
            chunk_terms = set(self._tokenized[index] if self._tokenized else _tokenize(self.chunks[index].text))
            exact_overlap = len(query_terms.intersection(chunk_terms))
            adjusted_scores.append(float(score) + exact_overlap * 0.25)

        top_indices = np.argsort(adjusted_scores)[::-1][:candidate_k]
        return {
            self.chunks[index].chunk_id: (self.chunks[index], float(adjusted_scores[index]))
            for index in top_indices
            if adjusted_scores[index] > 0
        }

    def _merge(
        self,
        query: str,
        dense: dict[str, tuple[Chunk, float]],
        sparse: dict[str, tuple[Chunk, float]],
    ) -> list[RetrievalResult]:
        max_dense = max((score for _, score in dense.values()), default=1.0)
        max_sparse = max((score for _, score in sparse.values()), default=1.0)
        target_section = infer_section(query)
        target_entities = set(infer_entity_type(query).split(",")) - {"general"}
        merged = []

        for chunk_id in set(dense) | set(sparse):
            chunk = dense.get(chunk_id, sparse.get(chunk_id))[0]  # type: ignore[index]
            dense_score = dense.get(chunk_id, (chunk, 0.0))[1] / max_dense if max_dense else 0.0
            bm25_score = sparse.get(chunk_id, (chunk, 0.0))[1] / max_sparse if max_sparse else 0.0
            score = self.dense_weight * dense_score + self.bm25_weight * bm25_score

            if target_section != "General" and chunk.section == target_section:
                score += 0.08
            if target_entities and target_entities.intersection(chunk.entity_type.split(",")):
                score += 0.06

            merged.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                )
            )

        merged.sort(key=lambda result: result.score, reverse=True)
        return merged

    def _rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.reranker_model)
            pairs = [(query, result.chunk.text) for result in results]
            scores = model.predict(pairs)
        except Exception:
            return results

        reranked = [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score) + math.log1p(max(result.score, 0)),
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                rerank_score=float(score),
            )
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked


def _section_filter(query: str) -> Optional[dict]:
    section = infer_section(query)
    return {"section": section} if section != "General" else None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9§.]+", text.lower())
