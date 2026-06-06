# PRD — Productizing FDA Traceability RAG as a Web Assistant (FarmersFront integration)

| | |
|---|---|
| **Status** | Draft for review |
| **Created** | 2026-06-06 |
| **About this repo** | A path to turn this demo (`RAG-fda-demo`) into an embeddable, production "Ask about FSMA 204" assistant |
| **Worked example consumer** | FarmersFront — an external Next.js landing page for FSMA 204 / Food Traceability compliance |

---

## 0. Why this doc lives here

This repo is a strong, **verified** local-first RAG over the FDA Food Traceability
Rule guidance. This PRD proposes how to productize it — wrap it in a service, host
it, and embed it in a real web product — using **FarmersFront** (a separate Next.js
FSMA 204 landing page) as the concrete first consumer. The intent is to give this
demo a credible productization path while keeping its core (ingestion, hybrid
retrieval, citation-aware generation) intact and reusable for any consumer.

The engine itself was cloned, built, and exercised end-to-end before writing this;
see [§12 Appendix](#12-appendix--evaluation-of-this-repo).

---

## 1. TL;DR

Add an interactive **"Ask about FSMA 204"** assistant to a web product. A visitor
types a plain-language question ("Is a farm with $20k in tomato sales exempt?") and
gets a grounded, **citation-backed** answer drawn only from official FDA Food
Traceability Rule guidance.

The retrieval engine already exists in this repo: hybrid dense + keyword RAG with
citation-aware generation and an out-of-scope guardrail. The work here is
**productization and integration**, not building RAG from scratch:

1. Wrap the Python pipeline in a small HTTP service (`/ask`, `/health`, `/status`).
2. Host it as a standalone microservice (it cannot run on Vercel/serverless).
3. Expose it through the consumer app (e.g. a Next.js route handler + chat widget).
4. Add the legal, freshness, and licensing guardrails a public-facing compliance
   tool requires.

FSMA 204 = 21 CFR Part 1, Subpart S = the "Food Traceability Rule" this repo
already covers, so its out-of-the-box content is directly usable for a farm-
compliance audience.

---

## 2. Background & context

**This repo (`RAG-fda-demo`).** A local-first hybrid RAG pipeline over the FDA Q&A
guidance and the Small Entity Compliance Guide. Ingestion prefers linked PDFs;
chunking is document-specific (Q&A pairs vs. paragraph windows); retrieval combines
Chroma dense vectors (`bge-base-en-v1.5`) with BM25 plus section/entity boosts;
generation is citation-aware with a scope guardrail (see [`README.md`](../README.md)).

**The worked-example consumer (FarmersFront).** A conversion-focused waitlist
landing page for FSMA 204 compliance and Walmart-ready traceability for small/
mid-size farms (separate Next.js 15 / React 19 / Tailwind / shadcn/ui repo). Its
audience already asks the exact questions this RAG answers (exemptions, CTEs, KDEs,
lot codes), at the point of highest intent.

---

## 3. Problem statement

Prospective users (small/mid-size produce farms) are anxious and confused about
FSMA 204: who is exempt, what records (KDEs) they must keep, what a CTE is, and what
retail buyers will require. Static FAQs can't answer a *specific* question. We want
to answer those questions **on-page, accurately, with citations** — without giving
legal advice or hallucinating regulatory requirements.

---

## 4. Goals, non-goals, success metrics

### 4.1 Goals
- G1. Free-text FSMA 204 question → grounded answer with citations to FDA source
  documents, embeddable in a web page.
- G2. Never present an answer not grounded in retrieved FDA guidance (refuse /
  deflect instead of guessing).
- G3. Usable as a conversion + qualification surface in the consumer app.
- G4. Keep the consumer app's deploy model (Vercel/static-friendly) intact — the
  heavyweight RAG runs as an external service.

### 4.2 Non-goals (this phase)
- Multi-turn memory / full conversational chatbot.
- FDA domains outside the Food Traceability Rule (drug, device, cosmetic, labeling).
- Authenticated / per-account compliance workflows.
- Replacing professional/legal compliance advice.

### 4.3 Success metrics
- **Engagement:** ≥ 15% of sessions interact with the assistant.
- **Conversion lift:** measurable signup increase for assistant users vs. not (A/B).
- **Answer quality:** ≥ 90% of a labeled question set judged faithful in offline eval.
- **Refusal correctness:** out-of-scope questions refused ≥ 95% of the time.
- **Latency:** p95 end-to-end answer < 6s (warm service).

---

## 5. Users & key use cases

| User | Question | What good looks like |
|---|---|---|
| Small produce farm owner | "Is a farm with $20k in tomato sales exempt?" | Correct threshold answer w/ § citation + CTA |
| Compliance-curious grower | "What is a CTE? What KDEs do I track?" | Plain-language definition grounded in guidance |
| Buyer-pressured supplier | "What records will a retailer require under FSMA 204?" | Scoped answer + product nudge |
| Off-topic visitor | "What are FDA rules on drug labeling?" | Polite refusal, redirect to in-scope topics |

---

## 6. Solution overview

```
 Consumer web app (e.g. FarmersFront — Next.js / React)
   │  chat card (shadcn/ui) placed near the FAQ
   │      └─ POST /api/ask  (same-origin)
   ▼
 Consumer server route handler  app/api/ask/route.ts
   │  - validates + rate-limits input
   │  - injects service auth, forwards to RAG service
   │  - never exposes the LLM/RAG keys to the client
   ▼
 RAG microservice  (FastAPI wrapper around THIS repo; separate host)
   │  POST /ask { question } -> { answer, citations[], mode, in_scope }
   │  - HybridRetriever (Chroma + BM25)       [retrieval.py]
   │  - AnswerGenerator (OpenAI-compatible)   [generation.py]
   │  - scope guardrail + citation contract
   ▼
 Prebuilt index (Chroma + chunks.jsonl), refreshed on a schedule
 LLM provider (OpenAI or DeepSeek, via OPENAI_BASE_URL)
```

**Why a separate service (not a serverless function):** this repo depends on
`torch` + `sentence-transformers` + `chromadb` (~2GB image, slow cold starts). That
cannot run on Vercel/Cloudflare edge/serverless. A small always-warm container is
the pragmatic split.

---

## 7. Detailed requirements

### 7.1 RAG service (thin HTTP layer over this repo)
- **FR-1** `POST /ask` accepting `{ "question": string, "top_k"?: int }`, returning:
  ```json
  {
    "answer": "string",
    "mode": "llm" | "extractive",
    "in_scope": true,
    "citations": [
      { "chunk_id": "qa_guidance:0007:…", "section": "Farms",
        "title": "…", "source_url": "https://www.fda.gov/…" }
    ]
  }
  ```
  This maps directly onto the existing `AnswerGenerator.answer(...)` output
  (`text`, `citations`, `used_llm`) in [`generation.py`](../src/fda_traceability_rag/generation.py).
- **FR-2** `GET /health` (index present, model loadable) and `GET /status` (chunk
  count, build time) for monitoring — `status` already exists in
  [`cli.py`](../src/fda_traceability_rag/cli.py) and can be reused.
- **FR-3** Ship a **prebuilt index** (image/volume); do not fetch fda.gov at request
  time. Re-ingest is an offline/scheduled job (`fda-rag ingest`).
- **FR-4** Generation provider configurable via env (`OPENAI_API_KEY`,
  `OPENAI_BASE_URL`, model) — verified to work with OpenAI **and** DeepSeek with no
  code change (the OpenAI SDK reads `OPENAI_BASE_URL`).
- **FR-5** Preserve the existing **scope guardrail** and **citation contract** in
  [`generation.py`](../src/fda_traceability_rag/generation.py).
- **FR-6** Stateless; horizontally scalable; bounded input length.
- **FR-7 (consideration)** Evaluate swapping local `sentence-transformers`
  embeddings ([`storage.py`](../src/fda_traceability_rag/storage.py)) for a hosted
  embedding API to shrink the image and cold start — only if it does not regress
  retrieval quality on the eval set (`fda-rag eval`).

### 7.2 Consumer integration (e.g. FarmersFront / any Next.js app)
- **FR-8** A server route handler proxies to the RAG service, injects the service
  token server-side, and applies per-IP rate limiting.
- **FR-9** A chat card built from the host app's UI primitives, placed near the FAQ.
- **FR-10** Render citations as links to the FDA source URLs.
- **FR-11** Persistent **"Not legal advice — based on FDA guidance"** disclaimer and
  a visible "draft guidance" note where applicable.
- **FR-12** Surface the app's primary CTA after an answer.
- **FR-13** Graceful degradation: if the service is down/slow, show a friendly
  fallback (never a stack trace).
- **FR-14** Suggested-question chips to lower the cold-start barrier.

### 7.3 Cross-cutting
- **NFR-1** No secrets in client bundle; all keys server-side.
- **NFR-2** Abuse protection (rate limit, max length, optional CAPTCHA if abused).
- **NFR-3** Log question + retrieved chunk ids + mode for offline quality review
  (treat as user content; privacy-reviewed).
- **NFR-4** Accessibility: keyboard-navigable, screen-reader labels, reduced-motion.

---

## 8. Phased delivery plan

| Phase | Scope | Deliverable | Rough effort |
|---|---|---|---|
| **P0 — Decisions** | Resolve licensing (see §10), pick host + LLM provider | Go/no-go | 0.5–1 day |
| **P1 — Service** | FastAPI wrapper (`/ask`, `/health`, `/status`), Dockerfile, prebuilt index, env-based provider | Deployable container | 2–3 days |
| **P2 — Deploy** | Stand up on Render/Railway/Fly; secrets; health checks; warm instance | Live service URL + monitoring | 1 day |
| **P3 — Frontend** | Route handler + chat widget + citations + disclaimers + CTA in the consumer app | Feature behind a flag | 2–3 days |
| **P4 — Hardening** | Rate limiting, abuse protection, graceful degradation, logging, a11y | Production-ready | 1–2 days |
| **P5 — Eval & launch** | Labeled question set, faithfulness + refusal eval, A/B wiring, launch | Metrics + GA | 1–2 days |
| **P6 — Ops** | Scheduled re-ingest, alerting, disclaimer/content review cadence | Runbook | 0.5 day |

**Estimated total:** ~9–13 engineering days to production; a clickable internal
demo (P1+P3 against a dev service) is reachable in ~3–4 days.

---

## 9. Technical decisions & trade-offs

- **Separate Python service vs. TS rewrite.** Chosen: separate service. Reusing this
  verified RAG is faster and lower-risk than rewriting hybrid retrieval in TS. Cost:
  another deployable to operate.
- **Local embeddings vs. hosted embedding API.** Start local (proven), measure,
  consider hosted to cut image size / cold start (FR-7).
- **Provider portability.** DeepSeek verified via `OPENAI_BASE_URL` with no code
  change; provider choice is a config/cost decision, not a code one.

---

## 10. Risks, legal & compliance

| Risk | Severity | Mitigation |
|---|---|---|
| **This repo has no LICENSE file** | **High / blocker** | Add an explicit license (or clarify intended terms) before any commercial reuse/integration. This is the top open item. |
| Presents *draft* guidance as final | High | Indexed Q&A is the Feb 2026 **draft**. Label clearly; prefer finalized guidance where available; re-ingest on updates. |
| Perceived as legal advice | High | Persistent "not legal advice" disclaimer (FR-11); scope to "general information from FDA guidance." |
| Hallucination / wrong compliance answer | High | Keep citation contract + scope guardrail; offline faithfulness eval gates launch; show sources. |
| Stale index | Medium | Scheduled re-ingest (P6) + `status` surfacing build date. |
| LLM cost / endpoint abuse | Medium | Server-side keys, rate limiting, input caps, cheaper model default. |
| Service downtime degrades host page | Medium | Graceful fallback (FR-13); widget is additive, never blocks the page. |

---

## 11. Open questions

1. **Licensing** of this repo — what terms allow reuse/integration? (Blocks P0.)
2. Production LLM provider — OpenAI vs. DeepSeek (cost, data handling, latency)?
3. Hosting target — Render / Railway / Fly / other?
4. Gate the assistant behind email capture, or keep it fully open?
5. Should answers stay neutral/informational, or tie back to the consumer's product?

---

## 12. Appendix — evaluation of this repo

`RAG-fda-demo` was cloned, installed, and exercised end-to-end before writing this:

- **Build/tests:** clean install (Python 3.12); 5/5 unit tests pass.
- **Ingest:** fetched the real FDA PDFs (`fda.gov/media/191182` Q&A draft Feb 2026;
  `fda.gov/media/168142` Small Entity Compliance Guide May 2023) → **193 chunks**
  (23 Q&A pairs + 36 Q&A context + 134 paragraph windows) with rich metadata.
- **Retrieval:** `search "TLC"` ranks "Traceability Lot Codes" chunks first; `eval`
  routes representative questions to correct sections (Farms / Shellfish / Critical
  Tracking Events).
- **Generation (LLM path, via DeepSeek):** the $20k-tomato exemption question
  produced a faithful answer citing § 1.1305(a)(1)(ii) and the guidance's own
  $20k-tomatoes + $20k-squash = $40k worked example, with real chunk-id + URL
  citations.
- **Guardrail:** "drug labeling" question correctly refused (`mode: extractive`,
  no LLM call).
- **Portability:** ran against DeepSeek unchanged via `OPENAI_BASE_URL`.

**Applicability to other FDA sources:** the fetch/storage/retrieval scaffolding is
generic, but domain knobs are hardcoded to the Food Traceability Rule
(`SECTION_KEYWORDS`, `ENTITY_KEYWORDS` in
[`chunking.py`](../src/fda_traceability_rag/chunking.py);
`DOMAIN_TERMS`/`OUT_OF_SCOPE_TERMS` in
[`generation.py`](../src/fda_traceability_rag/generation.py); the eval set in
[`cli.py`](../src/fda_traceability_rag/cli.py)). Reusable for traceability-adjacent
sources; an unrelated FDA domain requires editing those dictionaries (the scope
guard actively rejects drug/device/cosmetic/labeling today). Making it config-driven
/ multi-domain is a possible future enhancement, out of scope for this PRD.
