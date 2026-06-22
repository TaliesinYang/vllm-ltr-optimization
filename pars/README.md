# PARS contribution — files that apply onto the base fork (hao-ai-lab/vllm-ltr)

Our optimization (Dazhi's thread). Drop these into the fork's `train/`, plus the 3 trainer.py edits
documented in `../docs/PARS-PLAN.md`. Verified to import + syntax-clean 2026-06-22.

| file | -> goes in fork at | what |
|---|---|---|
| `marginRanking.py` | `train/allrank/models/losses/marginRanking.py` | PARS pairwise margin loss + delta-filter |
| `config_prefill_bert.txt` | `train/configs/config_prefill_bert.txt` | BERT-base backbone config (max_length=512) |
| `eval_gap.py` | `train/eval_gap.py` | Tau in-dist vs cross-dist (ShareGPT) = generalization gap |
| trainer.py edits (3) | `train/trainer.py` | import + `--margin/--delta` args + `marginRanking` loss branch |

Train PARS:  `python trainer.py --config configs/config_prefill_bert.txt --file <lmsys-train> --loss marginRanking --margin 1.0 --delta 0.2 --label-group-size 10 --tokenizer <local-llama> --run-id bert-pars-...-OURS`
Test gap:    `python eval_gap.py --usage-config MODEL/results/<run-id>/usage_config.json --in-file <lmsys> --cross-file <sharegpt>`
