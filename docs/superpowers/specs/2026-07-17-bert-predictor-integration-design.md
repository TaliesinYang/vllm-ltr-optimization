# BERT Predictor Integration Design

## Scope

Load the real `bert-prompt_schema-tier2-seed17` checkpoint from
`checkpoints_best_predictor/` on CPU, expose it through the existing
`Predictor` protocol, and make the `/v1/decision` development service select it
with `--predictor bert`. Keep the four policies, runner, and `ltr_training/`
unchanged.

## Authoritative Training Feature

The checkpoint was trained by the unmerged training worktree. Tier-2 loading
copies the source `prompt`, raw ToolACE `system` text as `tool_schema`, and
history into `TrainingExample`. For `variant="prompt_schema"`,
`train_ranker.render_example()` returns exactly:

```text
[USER]
{prompt}
[TOOLS]
{tool_schema}
```

The implementation must use the literal expression
`f"[USER]\n{prompt}\n[TOOLS]\n{tool_schema}"`. It must not insert history,
request metadata, manually-added BERT special tokens, or a JSON serialization
of the OpenAI `tools` list.

Training tokenization passes a list of rendered strings with `padding=True`,
`truncation=True`, `max_length=512`, and `return_tensors="pt"`. Inference uses
the tokenizer saved beside the checkpoint and the same arguments.

## Admission-Time Input

Serving-model token IDs are not BERT token IDs. `DecisionApplication` therefore
adds `prompt_text` and `tool_schema_text` to `PredictorInput.metadata`:

- `prompt_text` is the final message's non-empty string content;
- `tool_schema_text` is the single non-empty system message content.

The BERT predictor requires both strings. If either is unavailable, it raises a
clear `ValueError`; it never guesses a training feature from structured tools.
The existing serialized `prompt_token_ids` remain unchanged for other
predictors.

## Model and Prediction

`BertPredictor` loads `AutoTokenizer` and
`AutoModelForSequenceClassification` from a local directory with
`local_files_only=True`, moves the model to CPU, and calls `eval()`. Each
prediction runs under `torch.inference_mode()`.

The pairwise training loss assigns target `+1` when the left output is longer,
so a longer output must receive a higher raw logit. Validation computes
Kendall's tau directly between logits and output lengths and the checkpoint has
positive test tau `0.642329`. Scheduler cost is therefore:

```text
score = sigmoid(raw_logit)
```

This monotonic mapping preserves the learned ordering and maps it to `[0, 1]`:
shorter requests get lower scores and schedule earlier. It is not a calibrated
probability or empirical percentile.

Prediction latency is CPU wall-clock time from the start of feature rendering
through tokenization, forward inference, and score mapping. Checkpoint loading
is constructor work and is excluded.

## Honest Reliability Placeholders

Until ensemble calibration exists, every successful prediction returns a
constant `confidence=0.9`, explicitly documented as uncalibrated. Ensemble-based
confidence is future work. Until an evaluated OOD detector exists, `ood=False`,
explicitly documented as a placeholder. No logit magnitude or prompt-length
heuristic may be presented as calibration or OOD detection.

## Service and Smoke Path

`scripts/run_decision_service.py` gains `--predictor {stub,bert}` and
`--checkpoint`. Stub remains the default and retains existing flags. BERT uses
the real checkpoint and the `prompt_schema` feature variant.

The CPU smoke reads a real invocation from the pinned ToolACE snapshot, creates
one `PredictorInput`, prints the real prediction, then sends the same raw prompt
and system schema through a local `/v1/decision` HTTP server. Success requires
`reason_code="prediction_reliable"` and `prediction_reliable=true`.

## Tests

Add five focused unit tests covering exact rendering/tokenizer arguments, score
direction and honest placeholders, fail-closed missing raw features, exact
decision-service metadata extraction, and CLI stub/BERT selection. The real
417 MiB checkpoint is covered by the separate CPU integration smoke rather than
mocked into every unit test.
