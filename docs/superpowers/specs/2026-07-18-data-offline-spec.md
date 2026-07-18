# 数据与离线证据线 Spec（R1 冻结稿）

依据：课程库《手稿计划-2026-07-18.md》§2A/§2B 与《评审合并决策表-2026-07-18.md》。本地只做脚本与小规模验证；重推理在租用服务器执行。

## ARTIFACT 1：OOD 转换器

- 唯一 source pin 来源为主线 `configs/source-declarations.json`：BFCL `gorilla-llm/Berkeley-Function-Calling-Leaderboard@61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`；Toolathlon `hkust-nlp/Toolathlon-Trajectories@6194034105bc27fa438447172be0e7b4e35396e4`。
- 下载环境使用 `HF_ENDPOINT=https://hf-mirror.com`。
- 两域都只构造 first-assistant-invocation：BFCL 多轮只取 `question[0]`；Toolathlon 反序列化字符串字段后在首个 assistant 前截断。
- 输出独立 `LabelInput` 契约，不复用 `Tier1Label`。字段包含 `sample_id/request_id/prompt/tool_schema/history/session_id/task_id/source/source_revision/category`，且强制 `request_id == sample_id`。
- 各分层目标约 400 条，固定抽样 seed。manifest 至少记录 input/output SHA-256、`row_count/unique_task_count/unique_input_hash_count/unique_schema_hash_count`、转换错误及抽样 seed。
- 域切分只声明两级证据：`source-identity disjoint`；schema 是否重叠以 canonical JSON 排序键 SHA-256 实测，不声称天然零重叠。

## ARTIFACT 2：workload 构造器

- profile 支持 `id/ood/mixed`；ID 来自 ToolACE 测试切分，OOD 来自 ARTIFACT 1；mixed 接受固定 `ood_ratio` 与 seed。
- `baseline_service_ms = output_length * per_token_ms`，构造器接受 lengths 文件与 `--per-token-ms`；manifest 明确声明这是 proxy，不是 isolated service timing。
- 每条显式携带 `max_tokens=4096`、`category=id|ood/source`、`profile/source/source_revision/session_id/task_id/true_length`。
- `scheduler_benchmark/*` 在本批冻结，不做 runner 配合修改；manifest 将兼容状态记为未核验，若 runner 不接收字段则留待跨会话协调。
- stretch 提供 ShareGPT 聊天锚点转换器，默认固定 seed 抽样 300 条，同样只取 first-assistant-invocation，并使用相同 service-time proxy 与 `max_tokens=4096`；输入 revision 由调用方显式传入并随 manifest 落盘。

## ARTIFACT 3：三 seed 离线打分

- 固定 prompt_schema seed `17/42/73`，对 ToolACE test 1000 条与 BFCL/Toolathlon OOD 打分；逐条输出 `request_id`、三 seed raw score、三 seed 域内 percentile rank、`true_length/session_id/domain`、ensemble rank 与 dispersion。
- 每个 checkpoint 记录目录内容 SHA-256；缺任一 checkpoint 时写 typed blocked report，不启动推理。
- 分歧先在每域内转 percentile rank，再计算三 seed population standard deviation。
- `risk = 1 - Kendall tau-b`。输出名称固定为 `disagreement-empirical-error diagnostic`，不称 calibration/reliability。
- 每域报告 tokenizer 原始长度超过 512 的行数与截断比例。

## ARTIFACT 4：统计与泄漏核验

- Kendall variant 固定为 `b`；按每个预测列输出 point tau 与 95% percentile cluster-bootstrap CI，默认 1000 次并记录 seed。
- cluster 单元：BFCL=`id/task_id`；Toolathlon=`canonical task_name/task_id`；ToolACE=`session_id`。
- true_length ties 与 prediction ties 分开报告。
- 会话级与 tool-schema 级均使用 canonical JSON 排序键 SHA-256，分别报告 `train↔validation`、`train↔test`、`ToolACE(all splits)↔OOD` 三组交集。

## ARTIFACT 5：基线加固

- LightGBM 网格固定 20 组，覆盖 `max_depth/num_leaves/learning_rate/n_estimators`；仅用 validation 选最优，test 只评一次；报告范围、逐组 validation tau、最佳配置与 test tau。
- 旧聊天域五 checkpoint（listMLE-OPT、classification-OPT、PARS-BERT、A1、A2）为 `P2_optional`，不阻塞 R1。先提供 per-family backbone/head loader 描述、统一 `longer-is-higher` 方向反转与单测。
- 权重缺失或课程 vLLM/tokenizer runtime 不兼容时必须写 typed blocked，不做格式硬凑。

## 硬规则

- 不改模型、协议、已有标签文件或 `scheduler_benchmark/*`。
- 所有数字必须落 JSONL/JSON，禁止只在会话输出。
- 本批不 commit；R1 独立评审通过前不进入模拟器批次。
- 状态按 `ARTIFACT n: done/blocked-因为X` 汇报。
