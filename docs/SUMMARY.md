# Evaluation 总结 — vllm-ltr 复现 + PARS 优化

> CSCI 6806 Capstone · Deliverable #4 (Evaluation, 20%) · 测量日期 2026-06-22
> 硬件 RTX 4090 48GB / CUDA 12.1 · 模型 Meta-Llama-3-8B-Instruct · 单卡单 seed(42)
> **所有数字均为本人实测,无任何编造(Data Honesty)。**

---

## 1. 这个项目在做什么

复现并优化 **vllm-ltr**(Fu et al. NeurIPS'24,vLLM 0.4.1 fork):用一个轻量"输出长度预测器"给请求打分,在调度器里近似 **最短作业优先(SJF)**,从而减少 head-of-line blocking、降低延迟。

我们的贡献(3 条线):
1. **复现 baseline** —— FCFS vs LTR 延迟对比(不靠作者预训练,自己跑)。
2. **自训预测器** —— 复现 listMLE(排序)、训练 classification(分桶),并实现 **PARS**(Tao et al. 2025:pairwise margin loss + BERT backbone + delta-filter)。
3. **泛化性研究** —— 测 in-dist(LMSYS)vs cross-dist(ShareGPT)的 generalization gap,并 ablation 拆解 PARS 增益来源。

---

## 2. 做了哪些事(workflow)

| 阶段 | 内容 | 产出 |
|---|---|---|
| 环境 | 7 个修复(cmake/nvcc/transformers 降级/缺包/OOM→batch4 等) | `scripts/setup.sh` · `docs/ENV-NOTES.md` |
| Baseline | FCFS 与 LTR 各跑 rate 2/4/8/16/32/64 延迟 sweep | `baseline-results.tgz`(24 文件) |
| 训练 | 自训 listMLE / PARS / classification 三个预测器 | 5 套权重(含 ablation A1/A2) |
| Gap (E1) | `eval_gap.py`:in-dist LMSYS 留出尾 vs cross-dist ShareGPT 的 Kendall's Tau | Table E1 |
| Ablation (E2) | 拆 PARS 三要素(loss / backbone / filter)各训一个单因子变体 | Table E2 |
| 分类延迟 | tpt-class10 延迟 sweep(尝试) | **0 完成,serving bug,列为 limitation** |

---

## 3. 数据与结果

### Table 1 — Baseline 延迟:FCFS vs LTR(实测)
TTFT = time-to-first-token(排队延迟主导);TPOT = time-per-output-token。

| rate (qps) | FCFS TTFT (ms) | LTR TTFT (ms) | **TTFT 加速** | FCFS p99_TPOT | LTR p99_TPOT |
|---:|---:|---:|:--:|---:|---:|
| 2  | 50.5 | 103.2 | 0.49× | 28.1 | 44.2 |
| 4  | 95.6 | 121.6 | 0.79× | 62.0 | 65.5 |
| 8  | 137.1 | 203.1 | 0.68× | 97.9 | 160.0 |
| 16 | 17274.2 | 6043.7 | **2.86×** | 145.2 | 332.9 |
| 32 | 91581.5 | 48088.8 | **1.90×** | 158.9 | 715.4 |
| 64 | 258346.9 | 157546.3 | **1.64×** | 171.2 | 1400.3 |

**结论:** 高负载(rate ≥16,出现排队)下 LTR 把 **TTFT 砍到约 1/3(rate 16 = 2.86×)**,这就是 LTR 减少 head-of-line blocking 的核心收益;低负载下没排队可优化,预测器开销反让 LTR 略慢。代价:LTR 重排短作业优先 → 长作业等待,**p99_TPOT 变高**(典型 SJF 权衡,如实呈现)。

### Table 2 — 预测器质量:LTR vs 分类(Kendall's Tau)

| 预测器 | Backbone | Loss | **Tau** | 说明 |
|---|---|---|---:|---|
| classification | OPT-125M | crossentropy(10桶) | **0.194** | acc 0.965;离散桶 → 排序信号弱 |
| listMLE | OPT-125M | listMLE | **0.559** | 复现的排序 baseline |
| **PARS (ours)** | BERT-base | margin+δ | **0.596** | 最优 |

**结论:** Tau 0.194 ≪ 0.559/0.596 —— **预测器层面就证明了 base paper 选 LTR 而非分类是对的**。

### Table 3 — 泛化 gap(E1):in-dist vs cross-dist
两预测器都只在 LMSYS 训练;cross 测从未见过的 ShareGPT。

| 预测器 | Tau (LMSYS in) | Tau (ShareGPT cross) | gap |
|---|---:|---:|---:|
| listMLE | 0.559 | 0.315 | 0.243 |
| **PARS** | 0.596 | **0.361** | 0.235 |

**结论:** PARS cross-dist **0.315 → 0.361(+15% 相对)**,泛化更好;但两者都仍明显过拟合(掉 ~0.24),印证 base paper 承认的缺陷,PARS 缓解但未消除。

### Table 4 — Ablation(E2):PARS 的增益来自哪?

| 变体 | Backbone | Loss | δ-filter | Tau cross | Δcross |
|---|---|---|---|---:|---|
| listMLE base | OPT | listMLE | — | 0.315 | — |
| A1 +pairwise only | OPT | margin | δ=0.2 | 0.303 | **−0.012**(loss 没帮助) |
| A2 +BERT no filter | BERT | margin | off | **0.368** | **+0.065**(主因) |
| PARS full | BERT | margin | δ=0.2 | 0.361 | −0.007(filter 没帮助) |

**诚实结论:** PARS 的跨分布增益**几乎全部由 BERT backbone 贡献**,pairwise margin loss 和 delta-filter 在本单卡 8B 设置下没帮上忙。直接回答了"是不是只是 BERT?" —— 基本是。可能原因:per-batch(非全局)δ-filter、小 batch、单 seed、短 prompt。

---

## 4. 诚实 Limitation(必须在 deliverable 披露)

1. **分类延迟拿不到。** `tpt-class10` 调度器在 rate 4/8/16/32 全部 `Successful requests: 0`(请求提交但永不完成,benchmark 在空数组上 IndexError)—— fork 的 class 调度路径 serving bug。另:`--swap-space 100` 会让服务器在共享容器里被 OOM-kill(已修为 16,但修了启动仍 0 完成,瓶颈是调度器非启动)。**无任何分类延迟数被编造;分类仅在 Tau 层面与 LTR 对比。**
2. **batch 4(非 32)** —— 48GB 单卡 OOM 限制,paper 用 32,可能略压低 listMLE 的 Tau。
3. **δ-filter 是 per-batch 实现**(slate=batch),非 PARS 的离线全局 filter,语义等价但实现不同。
4. **单 seed、单卡、8B 模型** —— 数字是本人实测,样本规模有限。

---

## 5. 已保存的 Artifacts

**本地** `deliverables/04-evaluation/`:
- **5 套预测器权重**(model.safetensors + config + usage_config):
  `listmle-OURS`(239M)· `pars-OURS`(209M)· `class…-OURS`(239M)· `A1-opt125m-margin`(239M)· `A2-bert-margin`(209M)
- **原始延迟数据** `baseline-2026-06-22/baseline-results.tgz`(55M,24 文件:FCFS+LTR × 6 rate 的 .pt+.json)+ `RESULTS-summary.txt`
- **日志** `logs/class-train.log`(Tau 0.194 训练曲线)· `logs/class-sweep-0completion.log`(0完成证据)

**GitHub** `TaliesinYang/vllm-ltr-optimization`:
- `docs/RESULTS-E1-gap.md`(E1 gap + E2 ablation + 分类 limitation)
- `pars/`(marginRanking loss · eval_gap · BERT config)
- `scripts/`(setup · run_baseline · run_ablation · run_classsweep)

---

## 6. 周三 Presentation 要点(一句话版)

1. **复现成功**:高负载下 LTR 把 TTFT 砍到 ~1/3(rate 16 = 2.86×),代价是 TPOT 升高(SJF 权衡)。
2. **预测器 LTR ≫ 分类**:Tau 0.55-0.60 vs 0.19,印证 base paper 的选择。
3. **PARS 优化**:cross-dist Tau +15%,泛化更好。
4. **诚实 ablation**:增益主要来自 BERT backbone,不是 loss/filter —— 这是负结果但有价值,指向未来工作(全局 δ-filter、更大 batch)。
5. **Limitation**:分类延迟受 fork serving bug 阻塞,如实披露,未编造。
