# Direction decision — 2026-07-25 (evening session)

Resolves the decision graph from the 07-25 handoff. Inputs: schema-variability probe
(`probes/schema-variability-2026-07-25/`), full-text reads of P07 (UniBoost, ICML'26),
P09 (JITServe, NSDI'26 camera-ready), Park et al. 2026 IEEE Access survey, and a P03
citation-identity re-verification. PDFs archived in the course repo under
`materials/references/` (UniBoost, JITServe camera-ready, Park survey, PARS arXiv v3).

## ① Spine claim (SETTLED — revised 07-26 after E1/E2/E3, user ruling "(b)")

> **In shared serving queues carrying heterogeneous agent clients, reading the request as
> TEXT (prompt + tool schema) improves output-length ranking over the scalar features
> deployed today by +0.20 τ, and holds across Cold-Start Transfer strata (unseen tool
> combinations and fully unseen tools), where identity/lookup features take zero hits and
> collapse to the scalar baseline. The schema's specific contribution concentrates where
> schemas repeat — the regime real deployments live in (probe-verified constancy) — and is
> capped by the current 512-token truncation, a hypothesis E4's cached two-tower encoder
> tests. The Cold-Start stratification also gives the Reliability Gate its first
> evidence-based confidence signal, replacing the hardcoded 0.9 placeholder.**

Honest decomposition (E2 control, must accompany the headline): of the +0.2034 gap,
~79% (+0.160) comes from the text encoder itself (BERT prompt_only over LightGBM scalars)
and ~21% (+0.044) from adding schema text; on cold-start strata schema adds only ~+0.015.
Where schema earns its keep: seen-combination rows (n=45) — prompt_only craters to 0.239
while prompt_schema holds 0.601. Caveat: the decomposition bundles encoder-vs-trees with
text-vs-scalars in its first step; state that in one sentence, do not overclaim precision.

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

| # | Experiment | Status (07-26) | Outcome |
|---|---|---|---|
| E3 | LightGBM same-recipe 3-seed rerun | **DONE** (`d0ca541`) | Baseline is deterministic w.r.t. seed (std = 0.0000 by construction — no stochastic sampling in recipe); 0.4268 reproduced bit-exact; CI [0.391, 0.461]. "Single-seed" objection void. |
| E1 | Schema-hash / identity baseline | **DONE** (`064f9ed`) | Identity adds +0.008 (inside noise). Only 45/999 test fingerprints seen in train (1.26 rows/fingerprint) — lookup has nothing to look up. Forced categorical use → τ falls to 0.4131 (memorisation). Identity route dead. |
| E2 | Cold-start evaluation (two-subset; S1–S4 re-stratification pending, few sec CPU from saved scores) | **DONE** (`f3067ec`) | Text holds on unseen strata: 0.627 (unseen-combo, n=954) / 0.639 (strict unseen-tools, n=333) vs hash/scalar 0.40–0.49, CIs separated. Pre-registered survival criterion PASSED. Control finding: 79/21 decomposition (see spine claim). Checkpoint provenance reproduced to delta 0. |
| E4 | **Cached two-tower encoder** — now dual-purpose: (i) per-request latency with schema precomputed, at the Decision Service contract; (ii) tests whether un-truncating the schema (median prompt+schema = 781 tok vs 512 cap) recovers schema-specific τ | pending | Kill for deployability leg: per-request cost still ≥ heuristic budget by orders of magnitude (JITServe's 7 ms QRF is the published bar) |
| E5 | Serving-level validation under a REAL agent trace (mean E2E / goodput, NOT P99 supremacy), replaying OpenCode-through-gateway traces | pending — **GPU rental only after E4 + live-chain trace collection** | ranking gain doesn't move mean/goodput even in heterogeneous queue |

Pre-registered criteria used for E1/E2 (ruled before results were known): primary =
session-clustered bootstrap 95% CI separation on unseen strata (community-standard);
secondary = Δτ ≥ 0.05 material-effect bar (self-imposed, stated as such); tie on seen
stratum = Δτ < 0.02 (expected, supports lookup-table deployment story, does not kill).

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
