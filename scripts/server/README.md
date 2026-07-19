# 恒源云租卡日脚本

本目录实现 rental-readiness v4 Task 6。脚本默认使用 `/hy-tmp/ltr` 持久目录，所有大文件由 OSS 恢复；独立 launcher 只捕获 `INT/TERM`，健康检查成功后解除 trap，因此 launcher 正常退出不会杀掉刚启动的服务。只有 `run_matrix.sh` 这类编排器持有 `EXIT` cleanup。

## Mac 一次性准备（提交后由人工执行）

1. 只准备并打包 ID workload v2；`mixed.v2.jsonl`、`ood.v2.jsonl` 及其 manifests 必须等租卡日 OOD 真打标完成后在服务器生成。
2. 登录恒源云 OSS CLI。
3. 设置 `MIXED_WORKLOAD`、`OOD_WORKLOAD`、`WORKLOAD_MANIFEST_DIR`、`OSS_PREFIX` 后执行 `inventory_and_repack.sh`。

该脚本核对 T7 上两个真实 tar，把三个 seed 的 `final/` 原子规范化到 `extracted/checkpoints_best_predictor{,_seed42,_seed73}`（但不把它们重复塞进 benchmark bundle），对 ledger 做 6000 latest-ID 与 5997 ok/3 error 盘点，并生成只含 ledger/reference/workload 的 bundle。脚本从 `tar -t` 输出提取两个既有归档的真实成员路径，再把这些路径、原始 bytes SHA-256 与真实字节数原子更新到 `manifest/oss-objects.json`，随后上传并 `oss ls` 回读。提交态 manifest 的 size/sha/unpacks_to 是结构化 `null` 占位；未填真值时服务器 restore 会硬失败。

`tier2-toolace-sample-6000.jsonl` 不在 T7 归档中，不能在 Mac 阶段伪造。租卡日 restore 会先从 `oss://lmcache-labels.tar.gz` 定位 `toolace-6bda777-qwen35.jsonl`。若 OSS 对象或 tier1 文件不存在，脚本只能走固定 fallback：下载 `Team-ACE/ToolACE` dataset revision `6bda777c88d21e5a204703c1ee45597a8fa4f734` 的 `data.json`，校验 SHA-256 `ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e`，再调用仓库 `scripts/extract_tier1_labels.py`；tokenizer 与 generator 配置继续由 committed `configs/training_sources.json` 固定，不接受任意命令注入。随后 `rebuild_tier2_sample.py` 强制校验：

- tier1 SHA-256 `6dc808aa8f76a5391d33c22ecb0ae2a2967d01c923c71ec85d84ec537e5f227b`
- sampling seed 42
- reference manifest 内完整 `sample_ids`
- rebuilt SHA-256 `ee5a5889ca3d9bbee7790e7a408bd1664a285b6410b4fee54e45786d3eecb709`

任何不一致都是 NO-GO。

## 租卡日顺序

1. clone 本仓到 `/hy-tmp/ltr/repo`，执行 `setup_env.sh`。
2. 执行 `restore_from_oss.sh`。它使用 `.partial-*` 下载、校验、原子就位，解包到内容寻址目录并更新 `artifacts/current`，重建 sample，并对三个 checkpoint 做 CPU 前向 smoke。
3. 临时以 stock scheduler 和 Tier-2 的五个输出形状参数启动 vLLM，执行 `PREPARE_QUANTILES=1 restore_from_oss.sh`：resume replay 只补未成功的 3 行，`merge_quantile_labels.py` 按文件顺序 latest-wins，且必须 exactly 6000/all ok/output_length，之后以 `--expected-count 6000` 构建 mapper。
4. 保持该打标 vLLM 运行，执行 `build_server_workloads.sh`：下载两个 pinned OOD source、各以 sample-size 400/seed 17 转换，并对完整 800 条 OOD 输入执行 direct vLLM resume labeling。标签和 combined-lengths 资产始终保留全量；benchmark workload 再以 seed 42 从 ID test 池 1000 条和 OOD 池 800 条确定性抽取：默认 mixed=`150 ID + 150 OOD`，OOD=`200 OOD`。目标可用 `MIXED_ID_TARGET`、`MIXED_OOD_TARGET`、`OOD_WORKLOAD_TARGET` 覆写，manifest 记录 seed、池大小、目标计数和输入输出 SHA-256。任一行数或 SHA 不一致即 NO-GO。完成后停掉临时 vLLM。该步骤必须在 saturation calibration 之前完成。
5. 执行 `launch_decision.sh`，再执行 `measure_decision_latency.sh`。后者先并发 8 warm 20 次（不计），再并发 8 测 200 次真 HTTP `/v1/decision`，写 raw samples 和 `max(2000, ceil(1.25*p99))`。
6. 执行 `build_gateway.sh`（固定 `manifest/gateway-pin.txt`），再执行 `launch_gateway.sh`。
7. 启动 stock vLLM 后执行 `calibrate_saturation.sh`，得到 `runs/calibration/capacity.json`。每个固定 grid 输出及其精确 `.runs` 目录都会先安全重建，calibration 不使用 `--resume`；第一档已饱和或全部未饱和都会失败。
8. 租卡前以 `CAPACITY_RPS=0.3` 和请求数运行 `compute_rental_budget.sh`；租卡日用 calibration 真值复算。预算读取 workload 的实际 300/200 默认行数，并与执行端共享 `MIXED_REPEATS`、`OOD_REPEATS`（默认均为 3）以及 7/4 个实际策略档数；另显式计入默认 25 分钟的 `ood_labeling_800_direct_vllm`。超过 5.25 小时硬失败，45 分钟只保留作重试余量。
9. 执行 `run_matrix.sh`。它先强制 vLLM import、协议接缝零 skip、双请求真链路可靠性门槛，再按 `MIXED_REPEATS` 跑 mixed 7 策略、按 `OOD_REPEATS` 跑 OOD 4 策略、stock-vs-shim parity gate 与成对 gateway overhead；budget gate 会硬校验 workload 行数、capacity、repeats 和策略档数全部与当前执行一致，parity 未过不会写 `DONE`。preflight 会清空本次精确 tag 以证明当前执行；每次 policy/overhead launch 则用 `time_ns` 生成不可复用的 attempt tag，runner 仍使用 `--resume`，但只有 exact scheduler/profile/steady90/repeats、当前 capacity/model/workload raw SHA/vLLM version，且 manifest/evidence 完整的输出才跳过启动。manifest 追加 `vllm_attempt_tags` 与状态而不覆盖历史 provenance；成功的 custom scheduler attempt 必须产生非空 order log。每个 attempt 结束后把 `order.jsonl`、`vllm.log` 与明确 run tag 复制进本次 `RUN_ROOT/vllm-evidence/<attempt-tag>/`；若脚本因 EXIT/INT/TERM 中断，active attempt 也会先安全停止、补归档并标为 failed。最后 tar、OSS 上传并回读。

协议接缝测试已于 2026-07-18 在 201（x86，`vllm==0.24.0` 实装）完成：`2 passed`。

所有 launcher 同时写 PID 与 `/proc/<pid>/stat` field 22 start-time sidecar。编排器停止 vLLM/decision/gateway 前会逐项验证 PID 为数字、start-time 未变化且 cmdline 含对应服务的固定签名；过期、复用或意外 PID 只删除 sidecar，不发送信号。

## 关键环境变量

常用覆写项：`LTR_ROOT`、`REPO_ROOT`、`VENV`、`OSS_PREFIX`、`RESULTS_OSS_URI`、`MIXED_WORKLOAD`、`OOD_WORKLOAD`、`CAPACITY_RPS`、`MIXED_ID_TARGET`、`MIXED_OOD_TARGET`、`OOD_WORKLOAD_TARGET`、`MIXED_REPEATS`、`OOD_REPEATS`、`OOD_LABELING_MINUTES`。`GATEWAY_FORK_URL` 默认 `https://github.com/TaliesinYang/VeloxMesh.git`。所有 runner 调用均使用真实 CLI flags：`--endpoint --model --workload --capacity-rps --scheduler-cls --output`，以及按场景需要的 `--api-key --scenario --load --profile --repeats --resume`。

vLLM launcher 固定复制五个输出形状参数：auto tool choice、`qwen3_coder` tool parser、`qwen3` reasoning parser、`enable_thinking=false`、model length 8192；刻意不复制 `max-num-seqs` 与 `gpu-memory-utilization` 这两个吞吐参数。
