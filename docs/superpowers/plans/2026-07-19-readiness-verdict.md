# 租卡日就绪终审判决(2026-07-19)

## 轮次:初审 NO-GO(3 P0)→ P0-1/2/3a 修复 → 复核发现 P0-4 → 右尺寸化+预算决策(用户:6h→8h)→ **最终 GO**

## 初审(NO-GO)
# 租卡日就绪终审判决书

## 最终判决

**NO-GO。**

核心实现大部分落地，但当前仍有 3 个会让租卡日流程直接中断的 P0：产物未发布、mixed/OOD workload 闭环缺失、预算 manifest 路径不一致。

## 11 项原始缺口复核

| # | 原始缺口 | 判定 | 核验结果 |
|---:|---|---|---|
| 1 | VeloxMesh 无真实 decision 通路 | ⚠️ 部分解决 | Go 注入链、两调用点、上游透传均存在；但固定 SHA 尚未 push，服务器无法 checkout。 |
| 2 | 客户端 xargs 无信任边界 | ✅ 已解决 | 仅接受 `ltr_kind/ltr_category` 字符串；估值、可靠性、decision_id 均由 gateway 重建。[decision.go](/Users/alex/develop/VeloxMesh/internal/ltr/decision.go:36) |
| 3 | decision 响应验证不足 | ✅ 已解决 | schema、decision_id、概率、OOD、reason、feature、estimate、mapper provenance 均严格校验，违约 fail-open。[decision.go](/Users/alex/develop/VeloxMesh/internal/ltr/decision.go:74) |
| 4 | bool 破坏 vllm_xargs int 合同 | ✅ 已解决 | transport 输出 0/1；predictor 排除 bool；真实协议 seam 测试存在。[predictor.py](/Users/alex/develop/vllm-ltr-optimization/scheduler_benchmark/predictor.py:100) |
| 5 | 300 ms 超时无依据 | ⚠️ 租卡日待证 | 默认 2000 ms 已落实；并发 8、warm 20、采样 200、1.25×p99 脚本正确。尚无 GPU 日实测 manifest。[measure_decision_latency.sh](/Users/alex/develop/vllm-ltr-optimization/scripts/server/measure_decision_latency.sh:15) |
| 6 | 请求形状偏离 Tier-2 | ✅ 代码解决 | chat completions、history、4096、thinking=false、tools/tool schema、SSE usage 均已迁移。 |
| 7 | train/server torch pin 冲突 | ✅ 已解决 | train/server 依赖已拆分；server 固定 vLLM 0.24，不直接 pin torch。 |
| 8 | OOD wrapped-tools 不兼容 | ✅ 已解决 | 已识别 OpenAI wrapped tools 并原样透传。[tier2.py](/Users/alex/develop/vllm-ltr-optimization/ltr_training/tier2.py:31) |
| 9 | 6000 labels/quantile 输入与 provenance 不闭环 | ⚠️ 安全阻断中 | sample SHA 与 latest-wins merge 正确；当前实测仍精确拒绝 3 条 error，未伪造 6000。[merge_quantile_labels.py](/Users/alex/develop/vllm-ltr-optimization/scripts/server/merge_quantile_labels.py:55) |
| 10 | FQCN、重排审计、parity/overhead 门槛缺失 | ✅ 代码解决 | 7 档 mixed、4 档 OOD、order log、FCFS parity、paired overhead 均已接入。 |
| 11 | 不可移动交付、OSS、6 小时预算闭环 | ❌ 未解决 | 两仓未 push、bundle 未上传、mixed/OOD workload 缺失，且预算文件路径存在代码阻断。 |

结论：**7 已解决、3 部分解决、1 未解决**；其中部分项包含租卡前必须清除的 P0。

## 8 个 Task 完成度

| Task | 判定 | 完成度 |
|---:|---|---|
| 1 Go adapter | ⚠️ 部分完成 | 实现、测试、固定 SHA 均存在；但 VeloxMesh 本地仅有 `origin=zardonc`，目标分支无 upstream，Task 要求的 fork push 未完成。 |
| 2 Decision service | ✅ 完成 | contracts、4096、tool schema、mapper、manifest bytes SHA、可靠/不可靠 provenance、CLI required gate 均存在。 |
| 3 Gateway predictor | ✅ 完成 | int 严格解析、bool 回退、`LTR_PREDICTOR=gateway`、单请求与重排日志均存在。 |
| 4 Tier-2/依赖拆分 | ✅ 完成 | Tier-2 文件、wrapped tools、train/server requirements 已落地。 |
| 5 Runner/workload v2 | ⚠️ 部分完成 | runner 迁移完成，ID workload 1000 行有效；计划需要的 `mixed.v2.jsonl`、`ood.v2.jsonl` 均不存在。 |
| 6 服务器脚本/OSS | ❌ 未完成 | 脚本齐全且 ShellCheck 通过，但存在两个代码级 P0；bundle 仍标记 pending upload。 |
| 7 双层冒烟 | ⚠️ 基本完成 | CPU capture 实测 20/20、19 reliable、0 bool、0 schema 泄漏；201 的 `2 passed` 仅有用户声明，未按计划写入 README。 |
| 8 Fresh 重审 | ✅ 已执行 | 本判决即 fresh 终审；结果为 NO-GO。按用户要求未写文件。 |

## 新发现阻断

### P0-1：预算 gate 默认路径互不相容

预算脚本默认写：

- [compute_rental_budget.sh:8](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:8)：`/hy-tmp/ltr/rental-budget.json`

矩阵脚本只读：

- [run_matrix.sh:402](/Users/alex/develop/vllm-ltr-optimization/scripts/server/run_matrix.sh:402)：`$RUN_ROOT/rental-budget.json`

README 没有要求用同一个 `RUN_TAG` 覆写 `OUTPUT`。按文档顺序执行，矩阵会稳定报 `rental-budget.json missing`。

### P0-2：mixed/OOD workload 没有交付闭环

- README 要求 Mac 阶段先生成：[README.md:7](/Users/alex/develop/vllm-ltr-optimization/scripts/server/README.md:7)
- repack 实际只接收 ID workload：[inventory_and_repack.sh:8](/Users/alex/develop/vllm-ltr-optimization/scripts/server/inventory_and_repack.sh:8)
- 脚本又声明 mixed/OOD 在服务器生成，但没有对应服务器命令：[inventory_and_repack.sh:22](/Users/alex/develop/vllm-ltr-optimization/scripts/server/inventory_and_repack.sh:22)
- bundle manifest 也不含两文件：[oss-objects.json:39](/Users/alex/develop/vllm-ltr-optimization/scripts/server/manifest/oss-objects.json:39)
- 矩阵启动前却强制要求两文件：[run_matrix.sh:399](/Users/alex/develop/vllm-ltr-optimization/scripts/server/run_matrix.sh:399)

本地 repo、T7 与 bundle 中均未找到 `mixed.v2.jsonl` 或 `ood.v2.jsonl`。

### P0-3：服务器目前无法取得两仓完成态

- 主仓 `main` 相对 `origin/main` ahead 14。
- VeloxMesh 目标分支无 fork remote/upstream。
- 服务器构建脚本会从 GitHub clone 后 checkout 固定 SHA：[build_gateway.sh:20](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_gateway.sh:20)，该 SHA 当前未发布。
- bundle 明确标记 `pending upload after oss login`：[oss-objects.json:50](/Users/alex/develop/vllm-ltr-optimization/scripts/server/manifest/oss-objects.json:50)
- restore 会对所有对象先执行 `oss ls`，因此 bundle 未上传会立即失败：[restore_from_oss.sh:41](/Users/alex/develop/vllm-ltr-optimization/scripts/server/restore_from_oss.sh:41)

## 本轮新鲜验证

通过：

- 关键 Python 合同测试：`16 passed`
- `shellcheck scripts/server/*.sh scripts/smoke_local_chain.sh`：0 error
- Bash 语法、两仓 `git diff --check`：通过
- Python AST：84 files 通过
- bundle SHA：`d018e390...`
- sample SHA：`ee5a5889...`，6000/6000 unique
- tier1 SHA：`6dc808aa...`
- CPU capture：20 条、20 valid、19 reliable、0 bool、0 schema 泄漏
- 当前 merge 正确 NO-GO：精确指出 3 个缺失成功长度的 sample ID

本轮无法重新宣称：

- 完整 Python suite：只读沙箱无可写临时目录，pytest 在启动/collection 时失败。
- Go 全测：`go test` 无法创建 Go build 临时目录。
- 因此 `189 passed`、Go `123+`、201 `2 passed` 只能视为既有/用户提供证据，不是本轮新鲜复跑结果。

解除 NO-GO 的最低条件：修正预算路径；生成并打包/恢复 mixed、OOD workload 与 manifests；push 主仓 14 commits 和 VeloxMesh 固定 SHA；上传 bundle 并 OSS readback；随后执行 3 条 replay、生产 quantile、实测 timeout、真实预算、双请求门槛及 parity。

**最终判决：NO-GO；上述 P0 全部清除并完成 OSS/Git 可达性验证后，方可转 GO。**


## 复核一(P0-1/2/3a RESOLVED,新增 P0-4)
## 租卡日复核判决书

审核基线：主仓最新 `e60c6fb`，与 `origin/main` 完全一致；VeloxMesh `888fba9`，与 fork 分支及 `gateway-pin.txt` 完全一致。全程只读，未修改文件。

| 项目 | 判定 | 核验结论 |
|---|---|---|
| P0-1 预算路径 | **RESOLVED** | 生成端默认写 `$LTR_ROOT/rental-budget.json`，消费端读取同一路径：[compute_rental_budget.sh:8](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:8)、[run_matrix.sh:402](/Users/alex/develop/vllm-ltr-optimization/scripts/server/run_matrix.sh:402)。 |
| P0-2 workload 功能闭环 | **RESOLVED** | pinned OOD 下载、每源 400 条、direct-vLLM `nohup`/resume、latest-wins lengths 合并、mixed/OOD v2 构建和硬校验均已落地：[build_server_workloads.sh:148](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:148)、[build_server_workloads.sh:181](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:181)、[build_server_workloads.sh:211](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:211)。相关测试 `36 passed, 1 skipped`。 |
| P0-3a 两仓可达性 | **RESOLVED** | 主仓 `HEAD == origin/main == e60c6fb`；VeloxMesh `HEAD == fork/feat/ltr-decision-adapter == gateway-pin == 888fba9`；gateway clone 已使用 HTTPS。 |
| P0-3b bundle OSS 可达性 | **NOT-RESOLVED** | manifest 仍明确记录 `pending upload after oss login`：[oss-objects.json:50](/Users/alex/develop/vllm-ltr-optimization/scripts/server/manifest/oss-objects.json:50)。`e60c6fb` 增加了环境变量登录脚本，但 AK 配置、上传和 readback 尚无完成证据。 |

## 新发现的 P0 阻断

**P0-4：当前 workload 规模与 5.25 小时预算合同确定性冲突，NOT-RESOLVED。**

依据：

- OOD 固定为 `400 × 2 = 800` 条：[build_server_workloads.sh:17](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:17)、[build_server_workloads.sh:161](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:161)。
- ID test workload 为 1000 条；50% OOD 混合算法因此生成 1600 条 mixed workload：[workload-id-manifest.json:11](/Users/alex/develop/vllm-ltr-optimization/runs/workloads-v2/workload-id-manifest.json:11)、[workload_builder.py:54](/Users/alex/develop/vllm-ltr-optimization/ltr_training/workload_builder.py:54)。
- calibration 当前能够接受的最大 `capacity_rps` 是 1.5：若 2.2 首次饱和则选前档 1.5；若 2.2 仍未饱和则直接失败：[calibrate_saturation.sh:60](/Users/alex/develop/vllm-ltr-optimization/scripts/server/calibrate_saturation.sh:60)。
- 即便按最有利的 `capacity_rps=1.5` 计算，现有公式得到约 **12.303 小时**，远超 5.25 小时：
  - mixed 7×3：414.81 分钟
  - OOD 4×3：118.52 分钟
  - overhead：39.51 分钟
  - calibration、固定阶段、重启和上传：165.33 分钟
- budget 虽输出裁剪建议，但矩阵并没有消费该建议；mixed/OOD repeats 仍硬编码为 3：[compute_rental_budget.sh:62](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:62)、[run_matrix.sh:452](/Users/alex/develop/vllm-ltr-optimization/scripts/server/run_matrix.sh:452)、[run_matrix.sh:460](/Users/alex/develop/vllm-ltr-optimization/scripts/server/run_matrix.sh:460)。
- 新增的 **800 条 OOD direct-vLLM labeling** 也没有作为独立阶段计入 budget：[build_server_workloads.sh:185](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:185)、[compute_rental_budget.sh:38](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:38)。因此现有 12.303 小时还是低估值。

## 验证情况

- P0 相关 Python 测试：`36 passed, 1 skipped`
- VeloxMesh `internal/ltr`、`internal/providers/openai`：通过
- shellcheck、bash 语法检查：通过
- 全量 Python：`183 passed, 2 skipped, 7 failed, 2 errors`；失败均来自当前沙箱禁止监听本地端口或无法访问 Hugging Face。
- VeloxMesh 全量测试同样受端口绑定及 Redis/Qdrant/Postgres 缺失影响；LTR 和 OpenAI 目标包通过。
- 非阻断 P1：README 仍写 SSH fork 默认值，与实际 HTTPS 脚本不一致：[README.md:40](/Users/alex/develop/vllm-ltr-optimization/scripts/server/README.md:40)。

转为 GO 至少需要：

1. 让预算裁剪成为 `run_matrix.sh` 可执行配置，并确保计算端与执行端使用完全相同的策略数、repeats 和 workload 行数。
2. 把 800 条 OOD labeling 的实测或保守时间计入预算。
3. 使用最终 workload 与 calibration capacity 生成 `passed=true`、总计不超过 5.25 小时的 `rental-budget.json`。
4. 完成 bundle OSS 上传、`oss ls` readback 和 SHA 校验。

**最终判决：NO-GO（OSS bundle 尚未就位，且当前 workload/matrix 在最有利可接受容量下仍约需 12.303 小时，无法通过 5.25 小时硬门槛）。**
## 复核二(P0-3b/P0-4 右尺寸化 RESOLVED,最坏情形门槛遗留)
## 复核判决

当前主仓工作树干净，`HEAD == origin/main == 1061da3`。

| 项目 | 判定 | 依据 |
|---|---|---|
| P0-3b OSS 可达性 | **RESOLVED** | bundle 本地实际大小 `9,694,228` bytes、SHA-256 `d018e390…bf7e1` 与 manifest 精确一致；两个大 tar 的本地字节数也一致。dated results URI 已修正：[oss-objects.json:22](/Users/alex/develop/vllm-ltr-optimization/scripts/server/manifest/oss-objects.json:22)。restore 对三个主对象执行 `oss ls`，下载后校验 size 和 SHA-256：[restore_from_oss.sh:41](/Users/alex/develop/vllm-ltr-optimization/scripts/server/restore_from_oss.sh:41)、[restore_from_oss.sh:46](/Users/alex/develop/vllm-ltr-optimization/scripts/server/restore_from_oss.sh:46)。本会话尝试重复远端 `oss ls`，但沙箱禁止访问 `api.gpushare.com`；因此远端存在性采用已提交 readback 记录及用户提供的执行证据。 |
| P0-4 workload 右尺寸化 | **RESOLVED** | 默认 mixed=`150+150=300`、OOD=`200`，seed 42；完整 800 条 OOD 标签资产仍保留：[build_server_workloads.sh:14](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:14)、[build_server_workloads.sh:223](/Users/alex/develop/vllm-ltr-optimization/scripts/server/build_server_workloads.sh:223)。 |
| P0-4 预算/执行参数绑定 | **RESOLVED** | OOD labeling 25 分钟已计入；repeats、策略数、行数、capacity 均写入预算并由矩阵前置 gate 精确比较：[compute_rental_budget.sh:48](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:48)、[run_matrix.sh:416](/Users/alex/develop/vllm-ltr-optimization/scripts/server/run_matrix.sh:416)。在 `capacity=1.5` 下实跑得到 `5.08570454h`、余量 `9.8577min`，确实通过。 |
| P0-4 租卡前最坏情形 gate | **NOT-RESOLVED** | 计划及当前 README 仍要求租卡前以 `0.3 rps` 运行预算：[README.md:31](/Users/alex/develop/vllm-ltr-optimization/scripts/server/README.md:31)，脚本默认值也是 0.3：[compute_rental_budget.sh:9](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:9)。按当前 300/200、3 repeats、7/4 策略实跑结果为 **12.7400h、exit 1、passed=false**。即使执行既定裁剪顺序——OOD repeats 3→2、mixed 策略 7→4——仍为 **8.9892h**。仓库中也没有租卡日实测 `capacity.json` 可证明 1.5 是已校准真值。 |

验证结果：

- 相关 Python 测试：`11 passed`
- `shellcheck scripts/server/*.sh`：通过
- `bash -n scripts/server/*.sh`：通过
- Git diff/status：干净
- 用户报告的全量 `196 passed` 本会话未独立复跑，不作为否定依据

要解除最后阻断，必须二选一：

1. 继续缩小请求数/策略/repeats，使 `CAPACITY_RPS=0.3` 的租卡前预算真实通过；或
2. 正式修改计划，允许以有独立证据支持的容量下界替代 0.3，并提交该证据及对应预算。

**最终判决：NO-GO（P0-3b 已关闭，但 P0-4 仍未通过计划明确要求的租卡前 0.3 rps 最坏情形硬门槛）。**
## 终裁(GO)
## 最终复核

**P0-4：RESOLVED**

核验结果：

- `HEAD == origin/main == 35f6a5e`，工作树干净。
- 用户决策已写入权威计划与运行手册：[计划 Step 7b](/Users/alex/develop/vllm-ltr-optimization/docs/superpowers/plans/2026-07-19-rental-readiness-sprint.md:803)、[README](/Users/alex/develop/vllm-ltr-optimization/scripts/server/README.md:31)。
- 默认合同已改为 `CAPACITY_RPS=0.75`、gate `7.25h`：[compute_rental_budget.sh:13](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:13)、[compute_rental_budget.sh:69](/Users/alex/develop/vllm-ltr-optimization/scripts/server/compute_rental_budget.sh:69)。
- 原始吞吐 report 在 tar 与 T7 extracted 中的 SHA-256 完全一致；记录 `202.72 tok/s`、并发 8。`environment.json` 确认硬件为 RTX 3090。
- 完整 Tier-2 ledger 实测平均输出 `113.59 tokens`；按文档更保守的 130 tokens 计算：`202.72 / 130 / 2 = 0.7797 rps`，覆盖 0.75 下界。
- 默认参数独立实跑：
  - mixed 300、OOD 200
  - repeats 3/3、策略数 7/4
  - `total_hours=6.9992848`
  - `gate_hours=7.25`
  - gate 内余量约 `15.04min`
  - `passed=true`、exit 0
- 相关测试：`11 passed`；shellcheck 与 Bash 语法检查通过。
- 若租卡日 calibration 低于 0.75，现有 capacity 一致性 gate 会要求重新计算预算，裁剪顺序仍可执行。

非阻断文档尾项：计划开头 Goal 和历史复盘仍残留旧的“6h/5.25h”文字，但 Step 7b 与用户 2026-07-19 的明确决策已构成当前权威合同，不影响执行。

**最终判决：GO**