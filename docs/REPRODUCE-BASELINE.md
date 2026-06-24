# Baseline 如何复现 — 一步步(FCFS vs LTR 延迟对比)

> 对应论文 Fig.3(latency vs request-rate)。脚本:`scripts/run_baseline.sh`。
> 目标:在单卡 RTX 4090 上,自己跑出"高负载下 LTR 比 FCFS 延迟低"这条核心结论。

---

## 0. 一句话原理回顾

同一个 LLM、同一批请求,**只改调度顺序**:
- **FCFS** = 先来先服务(baseline,会队头阻塞)
- **LTR** = 用预测器估每个请求的输出长度,**预测短的优先**(近似 SJF)

跑同样 6 档负载,比两者延迟,差距就是 LTR 的收益。

---

## 1. 前置:环境 + 数据 + 模型

| 项 | 内容 | 怎么来的 |
|---|---|---|
| GPU 环境 | vLLM 0.4.1 fork + CUDA 12.1 | `scripts/setup.sh`(含 7 个修复:cmake≥3.21、nvcc PATH、transformers 降到 4.40.2、补 aiohttp/datasets 等、batch→4 防 OOM) |
| 模型 | Meta-Llama-3-8B-Instruct | 从 **ModelScope**(国内、免 gated token)下到 `/hy-tmp/models/` |
| 数据集 | `lmsys-…-c10000-…jsonl`(1万条真实对话,带"标准答案长度") | fork 的 Llama3-Trace,落在 `train/jsonfiles/` |
| 预测器(LTR 用) | `opt-125m-…-score-trainbucket10` 的 `usage_config.json` | 见 `MODEL/results/`(LTR 调度需要它来打分) |

> ⚠️ 注意:模型/数据盘是**临时盘**,跑完必须先 `collect_results.sh` 拉走结果再关机,否则数据丢失。

---

## 2. 核心机制:一个 `sweep()` 函数干三件事

`run_baseline.sh` 里每跑一种调度方法,就是这三步(简化自真实脚本):

```bash
sweep () {            # $1 = 调度类型(fcfs / opt-xxx)
  # ① 起 server,指定 --schedule-type
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --schedule-type "$1" \
    --enable-chunked-prefill --enforce-eager --port "$PORT" "$@" &
  sleep 90                                   # 等 8B 模型加载完(warmup)

  # ② 对 6 档负载逐个压测
  for r in 2 4 8 16 32 64; do
    python ../benchmarks/benchmark_serving_real.py --backend vllm \
      --model "$MODEL" --tokenizer "$MODEL" --dataset "$DATASET_FILE" \
      --num-prompts -1 --request-time 60 --schedule-type "$1" \
      --output-len -1 --request-rate "$r" --result-dir RESULTS --port "$PORT"
  done

  # ③ 关掉 server,换下一种方法
  kill $pid; sleep 60
}
```

**关键参数解释:**
- `--schedule-type` —— **整个实验的开关**:`fcfs` = 先来先服务;`opt-xxx` = LTR 调度(需配 `--prefill-predictor-model-config` 指向预测器)。
- `--request-rate r` —— 每秒发 r 个请求,r 越大负载越高(队头阻塞越明显)。
- `--request-time 60` —— 每档压测持续 60 秒。
- `--enforce-eager` —— 关掉 CUDA graph,单卡更稳。
- `--output-len -1` / `--num-prompts -1` —— 用数据集里的真实长度、跑全量。

---

## 3. FCFS vs LTR 的命令差异(就差一行)

```bash
# baseline:先来先服务
sweep fcfs   fcfs   --swap-space 16

# ours:LTR,多挂一个预测器 config
sweep opt-xxx opt-xxx --swap-space 32 \
      --prefill-predictor-model-config "$LTR_CFG"
```

唯一区别:LTR 把 `--schedule-type` 换成 `opt-xxx`,并喂一个**预测器**给 server。server 在请求入队时调用预测器打分,按"预测短的优先"重排。

> 📌 顺带:同脚本里 classification 那档用了 `--swap-space 100` —— 这就是后来分类 sweep 在共享容器里被 OOM-kill 的根因(见 `SUMMARY.md` limitation)。

---

## 4. 指标从哪来 → 怎么读

每次 `benchmark_serving_real.py` 跑完会打印并存一份结果(`RESULTS/*.json` + `*.pt`),关键两个延迟:
- **TTFT**(time-to-first-token)= 请求**排队 + 首 token**的延迟 → 队头阻塞主要体现在这。
- **TPOT**(time-per-output-token)= 生成阶段每 token 的延迟。

把 6 档汇总(`RESULTS-summary.txt`),就得到对比表(实测):

| rate | FCFS TTFT(ms) | LTR TTFT(ms) | **加速** |
|---:|---:|---:|:--:|
| 8  | 137 | 203 | 0.68×(低负载没排队,LTR 反略慢) |
| 16 | 17274 | 6044 | **2.86×** |
| 32 | 91582 | 48089 | 1.90× |
| 64 | 258347 | 157546 | 1.64× |

**怎么解读:**
- 低负载(rate ≤8):队列空,没队头阻塞可优化,预测器开销让 LTR 略慢 —— 正常。
- **高负载(rate ≥16):队列堆积,LTR 让短请求插队,TTFT 砍到 ~1/3(rate 16 = 2.86×)** —— 这就是复现出的核心收益。
- 代价:LTR 让长请求等待 → `p99_TPOT` 升高(典型 SJF 权衡,如实报)。

---

## 5. 复现 checklist(照着做就能重跑)

```bash
# 1. 环境(GPU box 上,一次性)
bash scripts/setup.sh

# 2. 确认模型 + 数据 + LTR 预测器 config 就位
ls /hy-tmp/models/Meta-Llama-3-8B-Instruct
ls train/jsonfiles/lmsys-*c10000*.jsonl
ls train/MODEL/results/opt-125m-*-score-*/usage_config.json

# 3. 跑 baseline sweep(tmux 里,约 30-40 分钟)
FORK_DIR=$HOME/vllm-ltr bash scripts/run_baseline.sh

# 4. 关机前务必拉走结果(临时盘!)
bash scripts/collect_results.sh
```

产出:`RESULTS/` 下 FCFS + LTR 各 6 档的 `.json`+`.pt`(共 24 文件)→ 已存 `baseline-2026-06-22/baseline-results.tgz`。

---

> 同目录:`BACKGROUND.md`(是什么/为什么)· `SUMMARY.md`(全部结果表)· `docs/RESULTS-E1-gap.md`(泛化 + ablation)
