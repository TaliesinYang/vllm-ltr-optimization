# vLLM LTR Optimization

Learning-to-rank scheduling for an LLM serving gateway: a fine-tuned ranker predicts a
request's output length, the scheduler orders the queue by that prediction, and a gate
withholds trust when the prediction should not be relied on.

## Language

### Scheduling

**Reliability Gate**:
The runtime decision of whether a given prediction is allowed to influence queue order.
_Avoid_: gate, confidence gate, OOD gate

**Fallback**:
Serving a request in arrival order because the Reliability Gate withheld trust.
_Avoid_: degrade, bypass, disable

**Run Check**:
A pass/fail assertion guarding a benchmark run, such as the parity, budget and preflight
checks. Unrelated to the Reliability Gate.
_Avoid_: gate, parity gate, budget gate

### Prediction

**Ranker**:
The fine-tuned BERT model that scores a single request's expected output length.
_Avoid_: predictor, model, BERT

**Decision Service**:
The HTTP service that wraps the Ranker and applies the Reliability Gate on behalf of the
gateway.
_Avoid_: predictor, predictor service

**Feature Variant**:
Which parts of a request appear in the Ranker's input text. The authoritative set is
`prompt_only`, `prompt_schema`, `full_context` — the names the nine trained runs carry.
_Avoid_: feature set, mode, and the abandoned `prompt` / `prompt_schema_history` /
`prompt_schema_history_workflow` naming

### Evaluation

**Ranking Tau**:
Kendall's tau between predicted and true output-length order, measured on a held-out split
of the same workload family the Ranker was trained on.
_Avoid_: tau, accuracy, correlation

**Cross-Workload Transfer**:
Ranking Tau measured on a workload family the Ranker was never trained on, such as a
chat-trained ranker scored against tool-calling traffic.
_Avoid_: OOD, generalization, distribution shift

**Ordering Tau**:
Correlation between served order and true order when the gate probe feeds deliberately
corrupted length hints. A property of the Reliability Gate, not of the Ranker.
_Avoid_: tau

**OOD Workload**:
The benchmark traffic drawn from BFCL and Toolathlon. Names a traffic source only; it
implies nothing about detection and is not the same concept as Cross-Workload Transfer.
_Avoid_: OOD

**OOD Detection**:
Deciding at runtime that an individual request falls outside the Ranker's training
distribution. A distinct concept from both OOD Workload and Cross-Workload Transfer.
_Avoid_: OOD
