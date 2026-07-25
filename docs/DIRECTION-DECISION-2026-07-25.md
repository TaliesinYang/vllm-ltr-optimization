# Direction decision — 2026-07-25 (evening session)

Resolves the decision graph from the 07-25 handoff. Inputs: schema-variability probe
(`probes/schema-variability-2026-07-25/`), full-text reads of P07 (UniBoost, ICML'26),
P09 (JITServe, NSDI'26 camera-ready), Park et al. 2026 IEEE Access survey, and a P03
citation-identity re-verification. PDFs archived in the course repo under
`materials/references/` (UniBoost, JITServe camera-ready, Park survey, PARS arXiv v3).

## ① Spine claim (SETTLED)

> **In shared serving queues carrying heterogeneous agent clients, the tool-schema text
> of a request is an untapped scheduling signal: it improves output-length ranking over
> the scalar tool features used by deployed schedulers (+0.203 τ offline), and — unlike
> per-client statistics — generalizes cold-start to unseen tool sets. Because the schema
> is constant within a deployment, its encoding is cacheable per client, which removes
> most of the predictor's per-request cost.**

Supporting acts (this is candidate (D) with B′ load-bearing):
- **Act 1 — workload characterization**: measured 68.9% of a real coding-agent request
  payload is tool schema (170 tools / 147 KB); schema constant within deployment+mode;
  ~1 in 4 requests per session carries zero tools (utility requests). External warrant:
  Park survey Challenge 4 explicitly calls for evaluation "under realistic agent
  workloads".
- **Act 2 — B′ predictor evidence**: schema text vs scalar features, with the new
  cold-start leg (experiment ladder below).
- **Act 3 — deployability/cost**: cached-schema encoding + confidence gate framed as
  *complementary* to JITServe (their §7 names "explicit fallback policies" as a gap).
  Gate is a guard, NOT the contribution.

### Why the old candidates died or got demoted
- **(A) "learned prediction improves serving latency" — RETIRED.** UniBoost beats
  *oracle* SRPT by 35.1% P99 TTLT (Table 2 — cite table, not the inconsistent abstract);
  no accuracy improvement answers that. Note: oracle SRPT still wins mean E2E by 19.1%,
  and UniBoost has zero agentic evaluation — cite honestly, scope its kill to tail claims.
- **(B) original phrasing — restated as B′.** Probe shows schema is byte-identical across
  turns/tasks within one deployment ⇒ per-request discrimination only exists in
  multi-tenant / mixed queues (which is what ToolACE's per-sample tools emulate).
- **(C) gate as spine — DEMOTED.** JITServe occupies "imprecise info usable online"
  with far stronger results (16×A100, within 3–9% of oracle; 7 ms QRF, BERT-class
  rejected at 56–187 ms under load). Residual gate claim is real but thin; our
  confidence value is still a hardcoded 0.9. Never claim superiority to JITServe.

## ② Hole order (SETTLED — and hole 2 is CLOSED)

Hole 2 first (it was free and could kill the root). Result: schema constant
within deployment ⇒ B must be multi-tenant + cold-start framed. Hole 1 ("τ didn't buy
latency") is re-scoped by P07: stop chasing tail-latency wins; the open question is
whether ranking helps mean/goodput under *mixed agent traffic* — that is E5, last rung.

## Experiment ladder (ordered by cost; each rung has a kill condition)

| # | Experiment | Cost | Kills B′ if |
|---|---|---|---|
| E1 | **Schema-hash baseline**: replace schema text with schema-identity categorical (+ scalar stats) on ToolACE; same recipe as BERT runs | $0, hours, 201 box | hash ties schema text on seen tool sets AND E2 shows no cold-start gap |
| E2 | **Unseen-tool-set split** (cold-start): schema text vs scalars vs hash on held-out tool sets | $0, hours | schema text ≤ hash/scalars on unseen split — B′'s distinctive leg dies |
| E3 | **Same-recipe confound close**: LightGBM 3-seed rerun so +0.203 isn't feature-type × model-family | $0, <1h | Δτ collapses when only feature set moves |
| E4 | **Cached-schema encoding prototype** (two-tower or schema-embedding lookup): per-request cost with schema precomputed; also fixes the 512-token truncation (median prompt+schema = 781 tok) | 201 box, ~1 day | per-request cost still ≥ heuristic budget by orders of magnitude → deployability act weakens (JITServe's 7 ms QRF is the published bar) |
| E5 | **Serving-level validation under mixed multi-tenant agent trace** (mean E2E / goodput, NOT P99 supremacy) | GPU rental — **only if E1–E4 survive** | ranking gain doesn't move mean/goodput even in heterogeneous queue |

**③ GPU: do not rent now.** E1–E4 are local. ④ backend choice: moot until E5.

## ⑦ Related Work positioning (unblocked)

- Prediction line: S³ (NeurIPS'23), LTR/Fu (NeurIPS'24), PARS, TRAIL, µ-Serve.
- **PARS citation fixed**: now "Ranking Before Serving …", ISC HP 2026, peer-reviewed,
  DOI 10.23919/ISC.2026.11520485. CSV patched 07-25. Local v2 PDF stale → v3 archived.
  "PARS" name survives in v3; gloss at first mention. 15.7× number is v3-only.
- Prediction-free attack: UniBoost — concede tail, keep mean + agentic-workload absence.
  Pre-empt its Fig. 2 noise-floor argument: B′ is *relative* ranking under shared noise.
- Imprecise-info SOTA: JITServe — complementary-gate framing, quote §7 fallback sentence;
  JITServe beats LTR-family on goodput (1.3–1.7× token) — the gate does not fix that;
  never imply it does.
- Framing: Park survey (FIFO HOL blocking = live problem; learned prediction still
  future-work as of Feb 2026; its silence is WEAK gap evidence — coverage is shallow).
- Standing caveats intact: no "first tool-aware scheduler" claim; BERT proxy prediction
  predates this project; Switchcraft's own future-work line stands.

## Independent items
- ⑤ Monday report: separate track; CLAIMS-AUDIT-2026-07-25.md + this doc are its inputs.
- ⑥ push `c858078`, close stale PR #2: pending user go-ahead.

## Full agent reports
Verbatim structured reports (P07, P09, survey, P03) live in the 2026-07-25 evening
session transcript; key numbers are reproduced above. BibTeX for all four targets is in
the reports and can be pasted into `latex_source/` when the report track needs it.
