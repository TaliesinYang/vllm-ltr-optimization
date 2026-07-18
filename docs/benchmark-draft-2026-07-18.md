# Benchmark 草稿 v2(2026-07-18,架构修正版)

> **架构定死(不再变):系统 = VeloxMesh gateway 为底座的完整服务栈。predictor、
> confidence gate、调度器、KV 管理——全部只是它的组件。所有 benchmark 都在这个
> 完整系统上跑,不存在「脱离 gateway 的跑法」。**

## 0. 为什么这样跑不影响归因

审稿人会问「你的提升是调度器带来的还是 gateway 带来的?」——答案是干净的:

```
四个策略(FCFS / pure-LTR / tail-safe / gated-hybrid)全部走同一条 gateway 路径:
    client → VeloxMesh → /v1/decision(predictor+gate)→ vllm_xargs → vLLM(scheduler)

→ gateway 开销对每个策略是同一个常数
→ 策略之间的延迟差异只能来自策略本身 → 公平对比成立
```

同一舞台上比四个演员,舞台成本相同,比较就公平。策略切换 = 换 `/v1/decision`
返回的决策逻辑 + 引擎内 scheduler policy,gateway 传输层不变。

## 1. 系统组件与被测配置

| 组件 | 归属 | 本轮状态 |
|---|---|---|
| Gateway 传输/路由 | VeloxMesh(Mingye,pinned `fc20873`)| 已有,Stage B smoke 7/13 通过 |
| `/v1/decision` 决策服务 | 本 repo | Codex #2 ARTIFACT 6(stub→真 predictor 换入)|
| Predictor(BERT + confidence)| 本 repo | 今日训练产出,周末换入 |
| 引擎内 scheduler(4 policy)| 本 repo,vLLM `--scheduler-cls` | Codex #2 已建 |
| 被服务模型 | Qwen3.5-9B BF16 pinned | 已部署 |

## 2. Benchmark 设计(单轨,全链路)

**环境**:租 GPU;vLLM 服务 Qwen3.5-9B BF16;VeloxMesh 在同机/LAN;predictor 跑
CPU、延迟计入端到端。所有请求走完整 gateway 链路。

**前置控制(系统内部,不改架构,各跑一次)**:
1. **FCFS parity**:stock vLLM 直连 vs 经我们 scheduler-shim(仍过 gateway),
   同 workload 同 seed,吞吐差 <3%、TTLT/TTFT 差 <5%(预定义)——证明自定义
   scheduler 代码路径本身不引入收益。
2. **Gateway 开销基线**:FCFS 直连引擎 vs FCFS 过 gateway,只测一次,给出 RPC
   开销的绝对数(诚实披露,一个数字,不进主对比)。

**负载**:先测 gateway 全链路的 FCFS 饱和吞吐,再按 40% / 70% / 90% 饱和度 +
1 个 burst 回放四档。

**Workload 三类**(各含 ID / OOD split):
- 单轮 tool-call(ToolACE)
- 多轮 workflow(Toolathlon)
- 长上下文 agent(lmcache canonical)

**对比策略(全部经 gateway)**:
- 全负载跑:FCFS / pure-LTR / tail-safe / **gated-hybrid(我们的)**
- 只在一个高负载点跑:oracle(长度真值上界)/ random(下界)/ 纯 aging
- gate 消融:always-on / 只 confidence / 只 opportunity / 双 gate

**指标**(主→次):
- 主:**TTLT P50/P95/P99、normalized slowdown、吞吐、starvation/最大等待**
- 次:TTFT、TPOT、predictor 覆盖率、**predictor+gateway 开销计入后的净收益**
- 每策略 gate 决策 log(reason/置信/采纳与否)——机制在真实路径生效的直接证据
- 统计:每配置 3 次重复,均值 + CI;所有策略同请求序列同 seed

## 3. 两轮 tool-call E2E(系统完整性)

在上面的 benchmark 之外,单独跑一次两轮 tool-call 走通:assistant 工具调用 →
工具结果回填(matching tool_call_id)→ 最终回复,全链路经 gateway。证明 gateway
保留 tool-call 语义 + 传输可靠预测。这是 spec §12 的完成标准,配架构图进报告。

## 4. 产出 → 报告六图

| 图 | 来源 |
|---|---|
| Fig.1 FCFS vs LTR 复现(2.86×)| 已有(midterm)|
| Fig.2 predictor 泛化 gap(chat 已有 + agentic ID/OOD)| 训练产出 |
| Fig.3 特征消融 + 结构基线 | Tier-1/Tier-2 矩阵 |
| Fig.4 selective risk-coverage | 训练产出(confidence)|
| Fig.5 模拟器 reliability envelope(实测 tau 校准)| Mac 重跑 |
| Fig.6 全链路 load sweep(四策略 P95/P99 TTLT,含开销)| §2 |
| (正文表)gate 决策路径 + gateway 开销 | §2 + §3 |

## 5. 时间线

| 日 | 动作 |
|---|---|
| 今天(五)| 6k 标签 → 微调矩阵 + 学习曲线(自动接)|
| 六 | 稽核 tau → 合并分支到 main → 真 predictor 换入 `/v1/decision` → 模拟器重跑 |
| 日 | 租 GPU + VeloxMesh 起链路 → parity + 开销基线 → 全链路四策略 benchmark → Fig.6 + 两轮 E2E |
| 7/21 | repo 清理 + latex_source/ + scripts/ + collaborator |
| 7/22 | 交 repo URL |
| 7/22-25 | 写正文(Alex,隊友 7/24 交小节)|
| 7/26 | rubric 逐项自查 |
| 7/27 8:59 | 交 PDF |

## 6. 与 Mingye 的合并

- 报告需要的「合并」= 接口冻结 + gateway 链路走通的证据,周末可单方面完成
  (他没空就用 pinned fc20873 自己跑,7/13 同款打法)。他有空则用其当前 main
  联调 30 分钟,证据更新鲜。
- 深度 E2E(他最新 main + 完整联调 + 延迟)= 9 月,架构不变只是升级版本。

## 7. 明确不做(7 月)

多模型家族(Gemma/gpt-oss)/ 真实 KV 抢占成本 / thinking 全因子 / API 旗舰迁移
/ 多 GPU 扩展——全部 future work。
