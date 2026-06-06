from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from fda_traceability_rag.models import SourceDocument


DEFAULT_SOURCES = {
    "qa_guidance": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/questions-and-answers-about-requirements-additional-traceability-records-certain-foods",
    "small_entity_guide": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/small-entity-compliance-guide-requirements-additional-traceability-records-certain-foods-what-you",
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    url: str
    doc_type: str


DEFAULT_SOURCE_SPECS = [
    SourceSpec("qa_guidance", DEFAULT_SOURCES["qa_guidance"], "Q&A"),
    SourceSpec("small_entity_guide", DEFAULT_SOURCES["small_entity_guide"], "compliance_guide"),
]


class SourceFetchError(RuntimeError):
    pass


def fetch_default_sources(timeout: int = 45) -> list[SourceDocument]:
    return [fetch_source(spec, timeout=timeout) for spec in DEFAULT_SOURCE_SPECS]


def fetch_source(spec: SourceSpec, timeout: int = 45) -> SourceDocument:
    response = _get(spec.url, timeout=timeout)
    content_type = response.headers.get("content-type", "").lower()

    if "pdf" in content_type or spec.url.lower().endswith(".pdf"):
        text = extract_pdf_text(response.content)
        title = _title_from_pdf_text(text) or spec.source_id
        return SourceDocument(spec.source_id, title, spec.url, spec.doc_type, text, spec.url)

    soup = BeautifulSoup(response.text, "html.parser")
    title = _extract_title(soup, spec.source_id)
    pdf_url = _best_pdf_link(soup, spec.url)

    if pdf_url:
        pdf_response = _get(pdf_url, timeout=timeout)
        text = extract_pdf_text(pdf_response.content)
        if len(text.strip()) > 500:
            return SourceDocument(spec.source_id, title, spec.url, spec.doc_type, text, pdf_url)

    text = extract_html_text(soup)
    if not text.strip():
        raise SourceFetchError(f"No usable text found for {spec.url}")

    return SourceDocument(spec.source_id, title, spec.url, spec.doc_type, text, spec.url)


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[Page {page_number}]\n{text}")
    return "\n".join(pages)


def extract_html_text(soup: BeautifulSoup) -> str:
    main = soup.find("main") or soup.find("article") or soup.body or soup
    for element in main(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    blocks: list[str] = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            blocks.append(f"\n{text}\n")
        else:
            blocks.append(text)
    return "\n".join(blocks)


def _get(url: str, timeout: int) -> requests.Response:
    headers = {
        "User-Agent": "fda-traceability-rag/0.1 (+https://www.fda.gov/)",
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _extract_title(soup: BeautifulSoup, fallback: str) -> str:
    h1 = soup.find("h1")
    if h1:
        title = " ".join(h1.get_text(" ", strip=True).split())
        if title:
            return title
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    return fallback


def _best_pdf_link(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    candidates = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        label = anchor.get_text(" ", strip=True).lower()
        absolute = urljoin(base_url, href)
        score = 0
        if "download" in href.lower() or href.lower().endswith(".pdf"):
            score += 2
        if "pdf" in label or "download" in label:
            score += 1
        if "/media/" in href:
            score += 1
        if score:
            candidates.append((score, absolute))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _title_from_pdf_text(text: str) -> Optional[str]:
    for line in _nonempty_lines(text):
        cleaned = re.sub(r"^\[Page \d+\]\s*", "", line).strip()
        if len(cleaned) > 20:
            return cleaned
    return None


def _nonempty_lines(text: str) -> Iterable[str]:
    return (line.strip() for line in text.splitlines() if line.strip())
