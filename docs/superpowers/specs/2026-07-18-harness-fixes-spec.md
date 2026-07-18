# Benchmark Harness 修理 Spec(5 项,TDD)

依据:课程库《手稿计划-2026-07-18.md》§2B 冻结版实验设计;file:line 为外部代码评审结论。

## ARTIFACT 1: 计时起点修正(防 coordinated omission)
- 延迟从「预定到达时刻」(replay_started + offset)起算,而非发送时刻。
- 现状:`scheduler_benchmark/runner.py:255` 在 sleep 后才 `perf_counter()`。
- 输出两口径字段:`scheduled_latency_ms`(主)与 `send_latency_ms`(对照保留)。

## ARTIFACT 2: 暖机丢弃
- 可配置 warmup(按请求数或比例,CLI 参数);summarize 仅统计稳态窗口。
- 现状:`runner.py:116-141` 全量样本入统计,无 warmup 概念。
- 结果 manifest 记录 warmup 配置与被丢弃样本数。

## ARTIFACT 3: 实验编排(可裁剪矩阵 + 断点续跑)
- scenario/负载档/重复数均 CLI 可选(单选/多选)。
- 现状硬编码:`benchmark_scenarios()` 固定 4 档(`runner.py:59`);`REPEAT_COUNT=3`(`:19`);`aggregate_repeats` 强制 ==3(`:147`);`T_CRITICAL` df=2 写死(`:20`)→ 重复数=5 会出错。
- 每个子 run(scenario×rep)完成即落盘独立文件;支持断点续跑(已完成的子 run 跳过)。现状 `:419-424` 全部跑完才写一次。
- 重复数任意(3/5/…)时统计聚合正确(t 临界值按实际 df 取或改报散点+区间说明)。

## ARTIFACT 4: prompt-length-SJF 策略
- 按 `prompt_token_ids` 长度升序调度,零预测器、零额外开销。
- 加入 `scheduler_benchmark/policies.py` 与 `vllm_scheduler.SCHEDULER_CLASSES`(现状仅 4 类,`vllm_scheduler.py:195`)。

## ARTIFACT 5: LTR+aging 独立策略
- pure-LTR 分数 + aging 项(参照 `policies.py:61` `_ltr_score` 已有 aging 逻辑),拆为独立可选策略,与 pure_ltr / gated 并列。

## 硬规则
- 不动 BertPredictor / decision_service / grpc_worker 核心逻辑。
- 不新增 lock/gate/contract。
- FCFS parity 校验逻辑保持可用(计时口径变更后 parity 语义仍成立,必要时 parity 同步用 scheduled_latency)。
- 全部测试绿(现有 106+);新功能各配测试。
- 进度按 "ARTIFACT n: done/blocked-因为X" 汇报。

## 验收
- `python -m pytest tests/ -q` 全绿。
- 演示命令:单独跑「混合档 90% × gated × 5 重复,warmup=100,断点续跑」一条 CLI 即可表达。
