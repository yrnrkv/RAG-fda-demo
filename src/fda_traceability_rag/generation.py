from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from fda_traceability_rag.retrieval import RetrievalResult


SYSTEM_PROMPT = """You answer questions about FDA Food Traceability Rule guidance.
Use only the retrieved context. If the context does not answer the question, say the
question may fall outside the covered FDA guidance documents. Cite chunk ids and
sections for every substantive claim."""

DOMAIN_TERMS = {
    "21 cfr part 1",
    "1.1305",
    "1.1315",
    "1.1325",
    "1.1330",
    "cte",
    "critical tracking event",
    "exemption",
    "farm",
    "fda traceability",
    "food traceability",
    "food traceability list",
    "foreign",
    "foreign firm",
    "fsma",
    "ftl",
    "harvesting",
    "initial packing",
    "kde",
    "key data element",
    "receiving",
    "retail food establishment",
    "restaurant",
    "shellfish",
    "shipping",
    "small entity",
    "traceability",
    "traceability lot code",
    "tlc",
    "transformation",
}

OUT_OF_SCOPE_TERMS = {
    "cosmetic",
    "device",
    "drug",
    "drug labeling",
    "labeling",
    "medical device",
    "pharmaceutical",
    "tobacco",
}


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    used_llm: bool


class AnswerGenerator:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    def answer(self, question: str, results: list[RetrievalResult]) -> Answer:
        citations = [_citation(result) for result in results]
        if not results or results[0].score < 0.08 or not _is_in_scope(question, results):
            return Answer(
                text=(
                    "This question may fall outside the covered FDA guidance documents. "
                    "Try asking about the Food Traceability Rule, exemptions, CTEs, KDEs, "
                    "TLCs, farms, retailers, restaurants, fishing vessels, or shellfish."
                ),
                citations=citations,
                used_llm=False,
            )

        if self.api_key:
            try:
                return Answer(
                    text=self._openai_answer(question, results),
                    citations=citations,
                    used_llm=True,
                )
            except Exception as exc:
                fallback = self._extractive_answer(question, results)
                return Answer(
                    text=f"{fallback}\n\nLLM generation failed, so this is an extractive answer. Error: {exc}",
                    citations=citations,
                    used_llm=False,
                )

        return Answer(text=self._extractive_answer(question, results), citations=citations, used_llm=False)

    def _openai_answer(self, question: str, results: list[RetrievalResult]) -> str:
        client = OpenAI(api_key=self.api_key)
        context = "\n\n".join(
            f"[{result.chunk.chunk_id}] Section: {result.chunk.section}; "
            f"Source: {result.chunk.title}\n{result.chunk.text}"
            for result in results
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nRetrieved context:\n{context}",
                },
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def _extractive_answer(self, question: str, results: list[RetrievalResult]) -> str:
        top = results[:3]
        lines = [
            "I found the most relevant FDA guidance passages below. Set `OPENAI_API_KEY` "
            "to enable a synthesized answer grounded in these passages.",
            "",
        ]
        for index, result in enumerate(top, start=1):
            chunk = result.chunk
            snippet = textwrap.shorten(chunk.text.replace("\n", " "), width=750, placeholder=" ...")
            lines.append(
                f"{index}. {snippet}\n"
                f"   Citation: `{chunk.chunk_id}`, section `{chunk.section}`, source `{chunk.title}`"
            )
        return "\n".join(lines)


def _citation(result: RetrievalResult) -> str:
    chunk = result.chunk
    return f"{chunk.chunk_id} | {chunk.section} | {chunk.title} | {chunk.source_url}"


def _is_in_scope(question: str, results: list[RetrievalResult]) -> bool:
    lowered_question = question.lower()
    question_terms = set(re.findall(r"[a-z0-9.]+", lowered_question))
    explicit_domain_hit = any(term in lowered_question for term in DOMAIN_TERMS)

    if any(term in lowered_question for term in OUT_OF_SCOPE_TERMS) and not explicit_domain_hit:
        return False

    if explicit_domain_hit:
        return True

    if not results:
        return False

    top_context = " ".join(result.chunk.text.lower() for result in results[:3])
    domain_hits = sum(1 for term in DOMAIN_TERMS if term in top_context)
    lexical_overlap = len(question_terms.intersection(set(re.findall(r"[a-z0-9.]+", top_context))))

    return domain_hits >= 3 and lexical_overlap >= 2
