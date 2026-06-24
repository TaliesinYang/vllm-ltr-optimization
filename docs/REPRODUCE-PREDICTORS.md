# 主线 B + C 如何做 — 预测器训练 / 泛化 gap / 消融

> 配套 `REPRODUCE-BASELINE.md`(主线 A)。这份讲:怎么训三个长度预测器(B)、怎么测泛化差距和做消融(C)。
> 训练/评估都在 `train/` 下,核心脚本:`trainer.py`(训练)、`pars/eval_gap.py`(gap)、`run_ablation.sh`(消融)。
> 训练 trace 统一用 LMSYS 切片 `c20000:30000`(和 base paper 一致),cross-dist 测试用 ShareGPT 同切片。

---

## 主线 B — 训练三个长度预测器

### B.0 通用机制(`trainer.py`,三种预测器共用)
所有预测器都在同一条训练管线里,只换 **backbone + loss + 标签方式**:

1. **标签 = 把"真实输出长度"映射成桶**(`RankingDataset.__len2label__`):
   `label = label_max//group_size − min(label_max,length)//group_size` —— **回答越短 → label 越大**(分越高 → 调度越靠前,对齐 SJF)。`--label-group-size` 控制桶粒度。
2. **三种 loss 分支**(`--loss`):
   - `listMLE` —— 排序损失(LTR baseline)
   - `crossentropy` —— 分类损失(把长度分到固定桶)
   - `marginRanking` —— **PARS 的 pairwise margin 损失(本人实现,`allrank/.../marginRanking.py`)**
3. 训练完把微调权重存到 `MODEL/results/<run-id>/finetuned/`(safetensors)+ `usage_config.json`,供调度器/评估直接加载。

### B.1 listMLE(LTR baseline,复现)
- backbone `facebook/opt-125m`,排序输出,桶粒度 10。
```bash
python trainer.py --config configs/config_prefill_opt.txt --file $TRACE \
  --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-score-trainbucket10-b4-OURS \
  --batch-size 4 --label-group-size 10 --loss listMLE --tokenizer $TOK
```
→ 结果 Tau ≈ **0.559**(in-dist)。

### B.2 classification(对照,验证"排序 > 分类")
- backbone `facebook/opt-125m`,**分类头**(`config_prefill_opt_classify.txt`:`mtype: class`),10 个桶(`group-size 820`,因为 8192/820≈10)。
```bash
python trainer.py --config configs/config_prefill_opt_classify.txt --file $TRACE \
  --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-class-trainbucket820-b4-OURS \
  --batch-size 4 --label-group-size 820 --loss crossentropy --tokenizer $TOK
```
→ Tau ≈ **0.194**(acc 0.965)。**远低于 listMLE/PARS,印证 base paper 选排序的理由**。

### B.3 PARS(优化,本人移植实现)
- backbone `bert-base-uncased`(`pars/config_prefill_bert.txt`:`mtype: rank`),**marginRanking** 损失,margin 1.0、delta-filter 0.2。
```bash
python trainer.py --config pars/config_prefill_bert.txt --file $TRACE \
  --job-dir MODEL --run-id bert-pars-llama3-8b-lmsys-margin1.0-delta0.2-b32-OURS \
  --batch-size 32 --label-group-size 10 --loss marginRanking --margin 1.0 --delta 0.2 --tokenizer $TOK
```
→ Tau ≈ **0.596**(in-dist),最高。
> PARS loss 核心(`marginRanking.py`):对每个有序请求对 `(高分, 低分)` 算 `max(0, margin − (s_high − s_low))`,并用 delta-filter 丢掉"长度差太小"的噪声对。关键修复:标签是 int64,做 −inf 掩码会溢出 → `.float()`。

---

## 主线 C — 泛化 gap + 消融

### C.1 Generalization gap(`pars/eval_gap.py`)
**怎么测:** 同一个预测器,在两个数据集上各算 Kendall's Tau:
- **in-dist** = 训练用的 LMSYS,取**留出的尾 10%**(训练时没用来更新,`held_out_tail=True`)。
- **cross-dist** = **完全没见过的 ShareGPT**。
- **gap = Tau_in − Tau_cross**,gap 越小越不过拟合。
```bash
python pars/eval_gap.py \
  --usage-config MODEL/results/<run-id>/usage_config.json \
  --in-file    jsonfiles/lmsys-...-c20000:30000-rFalse.jsonl \
  --cross-file jsonfiles/sharegpt-...-c20000:30000-rFalse.jsonl \
  --group-size 10 --tokenizer $TOK
```
对 listMLE 和 PARS 各跑一次 → 得 Table E1:listMLE cross **0.315**,PARS cross **0.361**(**+15% 相对**);两者 gap 都还有 ~0.24(过拟合仍在,PARS 缓解未消除)。
> 关键修复:微调目录没存 tokenizer → `tokenizer_name=cfg.model.pred_model`(从 base backbone 取)。

### C.2 消融(`run_ablation.sh`,定位 PARS 增益来自哪)
PARS 改了三样:**loss(listMLE→margin)、backbone(OPT→BERT)、delta-filter**。逐个隔离:

| 变体 | 隔离什么 | 命令要点(真实) |
|---|---|---|
| **A1** | 只换 **loss**(backbone 仍 OPT) | `config_prefill_opt.txt` + `--loss marginRanking --margin 1.0 --delta 0.2 --batch-size 4` |
| **A2** | 换 **BERT** 且 **关 filter**(`delta=0`) | `config_prefill_bert.txt` + `--loss marginRanking --margin 1.0 --delta 0.0 --batch-size 32` |

```bash
# A1 — isolate LOSS
python trainer.py --config configs/config_prefill_opt.txt --file $TRACE \
  --job-dir MODEL --run-id A1-opt125m-margin1.0-delta0.2-b4-OURS \
  --batch-size 4 --label-group-size 10 --loss marginRanking --margin 1.0 --delta 0.2 --tokenizer $TOK
# A2 — isolate BACKBONE (+filter off)
python trainer.py --config configs/config_prefill_bert.txt --file $TRACE \
  --job-dir MODEL --run-id A2-bert-margin1.0-delta0-b32-OURS \
  --batch-size 32 --label-group-size 10 --loss marginRanking --margin 1.0 --delta 0.0 --tokenizer $TOK
```
每个变体再跑 `eval_gap.py` 取 cross-dist Tau,对比得 Table E2 分解:

| 步骤 | cross-dist Tau | 贡献 |
|---|---:|---|
| base listMLE | 0.315 | — |
| + margin loss(A1) | 0.303 | **−0.012**(loss 没帮助) |
| + BERT backbone(A2) | 0.368 | **+0.065**(唯一功臣) |
| + delta-filter(full) | 0.361 | **−0.007**(没帮助) |

**诚实结论(关联非因果措辞):** 在本单卡 8B、单 seed 设置下,PARS 的跨分布增益**几乎只与 BERT backbone 同步出现**;margin loss 和 delta-filter 在此设置下未见收益。这是有价值的负结果,**不主张普适因果**,指向未来工作(全局 δ-filter、更大 batch、多 seed)。

---

## 复现 checklist(主线 B+C)

```bash
# 0. 环境同 baseline(setup.sh)。训练 trace + tokenizer 就位:
TRACE=jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl
TOK=/hy-tmp/models/Meta-Llama-3-8B-Instruct

# 1. 训三个预测器(B.1/B.2/B.3)→ 权重落 MODEL/results/<run-id>/
# 2. 跑消融 A1/A2(run_ablation.sh)
bash scripts/run_ablation.sh
# 3. 对 listMLE / PARS / A1 / A2 各跑 eval_gap.py → Table E1 + E2
# 4. 关机前 collect_results.sh 拉走全部权重(临时盘!)
```

产出:5 套权重(listMLE / classification / PARS / A1 / A2)+ E1/E2 表 → 已存 `deliverables/04-evaluation/`(权重)与 `docs/RESULTS-E1-gap.md`(表)。

> 同目录:`BACKGROUND.md`(是什么/为什么)· `REPRODUCE-BASELINE.md`(主线 A)· 本文(主线 B+C)· `SUMMARY.md`(全部结果)。
