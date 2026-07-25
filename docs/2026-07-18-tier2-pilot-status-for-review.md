# Tier-2 Pilot 当前状态与审核问题

> 给 Claude 的技术审核材料。状态快照时间：2026-07-18 01:06:49（UTC+8）。
> 本文只记录实测事实、已知故障与待决策项，不把 pilot 结果提前包装成成功。

## 1. 本次任务目标

在恒源云 RTX 3090 24GB 上完成：

1. Python 3.11 + 新版 PyTorch/vLLM 环境与完整测试；
2. Tier-1 训练配置 `save_steps: 10 -> 200`；
3. Tier-1 9 个 BERT run + 1 个 LightGBM run；
4. canonical LMCache Tier-1 全量标签；
5. 固定 revision 的 Qwen3.5-9B BF16 vLLM 三请求冒烟；
6. ToolACE Tier-2 pilot 400 条，随后仅在结果可信时考虑全量 ledger。

固定模型与推理配置：

- Model: `Qwen/Qwen3.5-9B`
- Revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Dtype: BF16；未量化
- vLLM: 0.19.1
- `--tool-call-parser qwen3_coder`
- `--reasoning-parser qwen3`
- `enable_thinking=false`
- `temperature=0`
- `max_tokens=4096`
- `max_model_len=8192`

## 2. Artifact 状态

### Artifact 1：环境完成

- Python 3.11.10
- PyTorch 2.10.0+cu128
- Transformers 5.14.1
- vLLM 0.19.1
- RTX 3090，CUDA 可用
- 最新远端完整测试：`33 passed, 14 warnings in 43.03s`
- 环境记录：`/hy-tmp/results/environment.json`

### Artifact 2：checkpoint 间隔完成

- 9 个 BERT 配置均使用 `save_steps=200`。

### Artifact 3：Tier-1 9+1 matrix 完成

汇总文件：`/hy-tmp/results/tier1-matrix-summary.json`，`completed_runs=10/10`。

| Run | Validation Kendall tau |
|---|---:|
| prompt_only seed17 | 0.620305 |
| prompt_only seed42 | 0.621654 |
| prompt_only seed73 | NaN |
| prompt_schema seed17 | 0.617369 |
| prompt_schema seed42 | 0.605513 |
| prompt_schema seed73 | 0.633744 |
| full_context seed17 | 0.615506 |
| full_context seed42 | 0.631434 |
| full_context seed73 | 0.638794 |
| LightGBM structural seed42 | 0.468186 |

`prompt_only/seed73` 的模型参数经检查为有限值，但 validation tau 为 NaN；未改 seed、未补造结果。其余 seed73 变体为有限 tau，因此不是 seed73 的全局运行故障。

### Artifact 4：LMCache Tier-1 完成

- 产物：`/hy-tmp/results/lmcache-6e043b9-full.jsonl`
- canonical source revision：`6e043b9e89865df3aec19fd5679286b683bfd70e`
- 权威数据源实际为 24,880 rows；原计划写 24,881，存在 1 条 source-count 偏差。
- SHA256：`ca6d9738e53447611403e2ea369acd083ec0ee4722083f8808dc10b9840709e7`

### Artifact 5：vLLM smoke 完成

三条真实 ToolACE 请求全部成功：

- completion tokens：57 / 121 / 322
- failure rate：0%
- censor rate：0%
- output throughput：32.68 output tokens/s
- 第一条 `finish_reason=tool_calls`，证明 tool parser 实际参与解析
- 报告：`/hy-tmp/results/vllm-smoke-toolace-3-report.json`

### Artifact 6：pilot 仍在运行

截至状态快照：

- 进度：233/400
- 成功 API 记录：233
- 失败：0
- censored：65
- censor rate：27.90%
- 累计输出 tokens：287,755
- 累计请求耗时：7,586.08 秒
- 端到端输出吞吐：37.93 output tokens/s
- 平均每请求：32.56 秒
- 输出长度 p50/p95/p99：141 / 4096 / 4096
- GPU：21,874 / 24,576 MiB，100% utilization，74°C
- ledger：`/hy-tmp/results/tier2-toolace-pilot-400.jsonl`
- log：`/hy-tmp/logs/tier2-toolace-pilot-400.log`

## 3. 当前主要问题

### 3.1 高 censor 不是 GPU 崩溃，而是模型大量生成到上限

从约第 169 条开始，多个相邻 ToolACE workflow 的请求连续生成满 4096 tokens：

- `finish_reason=length`
- `status=ok`
- 每条约 102-110 秒
- API/解析失败仍为 0

因此 GPU 100% 是有效解码负载，不是卡死。真正瓶颈是大量长序列的自回归解码。

这些 censored 样本在 session/workflow 上明显聚集。当前 pilot 直接取数据前 400 条，而非分层随机样本，因此 27.9% 是否代表全量总体分布尚未证明。

### 3.2 当前 ledger 没有保存生成正文

ledger 保存了长度、finish reason、usage、耗时和状态，但没有保存完整生成文本。因此目前不能直接区分：

- 模型是否进入重复循环；
- chat template/tool schema 是否诱发异常长自然语言；
- tool-call parser 是否只在部分 workflow 上失效；
- 数据本身是否要求长输出。

在直接扩全量前，应至少对若干 censored sample 做可复现的正文抽查。

### 3.3 D4 failure gate 与 censor 风险脱钩

当前 D4 仅判断 overall failure rate <= 1%。因为 censored rows 的 `status=ok`，即使 censor rate 很高，D4 仍会通过。

现有 watcher 在 pilot 完成且 D4 通过后，会自动复制 pilot ledger 并启动全量。也就是说，若不人工干预，27.9% censor 仍可能触发全量任务。

这不涉及新增代码 lock/gate；需要审核者判断是否应在当前实验决策层暂停全量。

## 4. 全量时间估算

当前实测平均耗时 32.56 秒/请求，约 110.6 requests/hour。

按当前总体平均值线性外推 13,819 条：

- 约 124.99 小时
- 约 5.21 天连续运行

pilot 完成后剩余约 13,419 条：

- 约 121.4 小时
- 约 5.06 天

这只是基于当前混合分布的估计。如果后续样本像最近的 censor cluster 一样，每条约 105 秒，则极端上界接近 16-17 天。

所以原先“挂一晚完成全量”的假设已被实测吞吐否定。

## 5. `max_tokens=4096` 是否合理

4096 不是临时拍脑袋设置，而是原定 pilot protocol。作为 pilot cap，它成功暴露了真实 censor 问题；但以当前 27.9% censor rate，不能直接证明它适合作为全量标签配置。

简单提高到 8192 也不是免费修复：

- 当前 `max_model_len=8192` 包含 prompt + completion；很多 prompt 已有约 700-2900 tokens，不能保证再生成 8192 tokens。
- 若模型在重复循环，提高上限只会约翻倍单条最坏耗时。
- 24GB 显存已经较紧，当前服务约占 21.9GB。
- 研究计划要求报告 2048 vs 4096 敏感性；不能在 pilot 中途偷偷换 protocol。

因此应先判断高 censor 的机制，再决定 4096、8192、停止条件或模板修正。

## 6. vLLM 启动故障与已验证修复

### 故障 A：CUDA graph/KV cache 初始化 OOM

原启动使用默认并发捕获规模，CUDA graph profiling 后 GPU 已占 23.30GiB，再申请 1.03GiB minimal KV cache 时 OOM。

最小修复：

- 显式 `--max-num-seqs 1`
- 理由：当前 replay client 本身严格串行，一次只有一个请求

修复后日志证据：

- GPU KV cache：22,176 tokens
- 8192-token request 的日志理论 concurrency：8.89x
- 服务 health ready
- 未量化、未换模型、未改 revision

### 故障 B：Hugging Face cache root 不一致

完整快照位于 `/hy-tmp/hf/models--Qwen--...`，但 vLLM 默认按 `HF_HOME` 查 `/hy-tmp/hf/hub/models--...`，后者是不完整副本。

修复：

- `HF_HUB_CACHE=/hy-tmp/hf`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

修复后 vLLM 明确把 model ID 映射到固定本地 snapshot，无重复 19GB 下载。

相关本地文件：

- `scripts/run_post_matrix_tier2.sh`
- `tests/test_post_matrix_tier2_script.py`

TDD 证据：新增测试先失败，再通过；最新远端完整测试为 33 passed。

## 7. 当前仍在运行的进程

- watcher：`bash /hy-tmp/staging/scripts/run_post_matrix_tier2.sh`
- vLLM server：Qwen3.5-9B pinned BF16
- pilot：`replay_tier2_labels.py --limit 400`

这些进程均为 nohup/脱离 SSH 状态。`/hy-tmp` 当前约 83GB 可用。

## 8. 请 Claude 重点审核

请不要只检查“进程是否正常”，而是回答以下研究与执行问题：

1. 27.9% censor 且集中在连续 workflow，是否足以暂停自动全量？
2. 当前前 400 条顺序 pilot 是否存在严重抽样偏差？应否改为 session/长度/schema 分层抽样后重跑 pilot？
3. 在不改变 pinned model/revision、不量化的前提下，最小诊断实验是什么？
4. 是否应先保存并人工检查 5-10 条 censored response 正文，判断重复循环、模板错误或真实长输出？
5. `max_tokens=4096` 是否应保留为主 protocol，并把 censored rows 排除主 loss；还是应先做小规模 2048/4096/可行上限敏感性？
6. 当前 D4 只约束 failure rate、不约束 censor rate，是否足以作为全量启动依据？
7. 以约 5.2 天的实测全量估算，是否值得继续本地全量，还是应改采样规模/并行客户端/更高吞吐硬件？
8. `prompt_only/seed73` 的 NaN tau 应视为真实训练不稳定性、评估退化，还是必须补做更深诊断？

## 9. 建议审核前不要做的事

- 不要把 censored 当 failure 后静默丢弃；
- 不要把 `max_tokens` 直接改到 8192 后全量重跑；
- 不要因为 D4 failure=0 就宣称 Tier-2 配置已通过；
- 不要改变模型 revision 或使用量化来掩盖当前问题；
- 不要编造全量可“一晚完成”的时间表。
