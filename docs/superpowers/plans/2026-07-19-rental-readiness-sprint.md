# 租卡日就绪冲刺 v4-final（Rental Readiness Sprint）

> 审核轨迹：v1 REJECT（架构级：VeloxMesh 无合同通路）→ v2 REJECT（15 项接口阻断）→ v3 REJECT（9 解/3 部分/3 未解 + 3 新发现）→ **v4 APPROVE-with-changes（7 项局部修正，已全部就地落实：ledger latest-wins 去重、不满 6000 即 NO-GO、fixture p10..p99+sample_count、provenance 精确校验+manifest 补 notice、协议接缝进 preflight 硬门槛+ood 判可靠、预算 gate 逐阶段计时、五参数定性为"形状相关"并声明吞吐参数不复制）**。本文即执行版。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通唯一真实链路 `runner → VeloxMesh(fork 分支) → decision service(BERT) → vLLM 0.24 自定义调度器`，使租卡日 6 小时 GPU 预算可以直接跑 6 策略 benchmark 矩阵。

**Architecture:** VeloxMesh 固定 commit `4b4b5ad` 上开 fork 分支 `feat/ltr-decision-adapter`：网关在转发前调用我们的 HTTP decision service，把预测注入上游请求体的 `vllm_xargs`；vLLM 0.24 自定义调度器通过 `sampling_params.extra_args` 读取（`vllm_scheduler.py:82-85` 已有通路）并用新的 `gateway` predictor 消费。runner 从 legacy completions 迁移到 chat completions（与 Tier-2 打标形状一致，否则 `true_length` 不可信）。

**Tech Stack:** Go 1.26.1（VeloxMesh，模块名 **`veloxmesh`**，见 go.mod:1——import 一律 `veloxmesh/internal/...`）、Python 3.11、vLLM 0.24.0 pinned、aiohttp、pytest、httptest（Go）。

## Global Constraints

- vLLM 固定 `vllm==0.24.0`（source revision `ee0da84ab9e04ac7610e28580af62c365e898389`）；runner 强制校验 `0.24.x`（`runner.py:615-616`）。
- 服务模型固定 `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，**`--dtype bfloat16`**，`enable_thinking=false`；HF 下载必须 `--local-dir /hy-tmp/models/qwen3.5-9b`。
- VeloxMesh 基线 `4b4b5ad9d34fdeb8b87f21981338498f25035d09`；Go 改动只进 fork 分支 `feat/ltr-decision-adapter`；**分支完成后记录最终 commit SHA，服务器 clone 按该 SHA checkout**（不能只锁可移动的分支名）；PR 由用户日后提交。
- Go 工具链：`GOTOOLCHAIN=auto`（go.mod 要求 1.26.1）。
- **vllm_xargs 类型合同：固定 revision 的协议值类型只允许 str/int/float/list，没有 bool**（bool 会被 coerce 成数字）。因此 `prediction_reliable` 在 xargs 里固定为 **int 0/1**；predictor 端按 `== 1` 严格解析。所有涉及该字段的测试都用 0/1。
- benchmark 请求形状与 Tier-2 打标一致：chat completions、`chat_template_kwargs={"enable_thinking": false}`、temperature 0、max_tokens 4096、**携带完整 history**（`tier2.py:69` build_request 会加入全部历史消息）。
- **vLLM serve 的五个"形状相关"参数必须复刻打标环境**（证据 `.worktrees/final-training-artifacts/scripts/run_post_matrix_tier2.sh:54`）：`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' --max-model-len 8192`。缺 tool parser 会改变 tool-call 回应形状与 finish reason，`true_length` 比较即失效。打标时的 `--max-num-seqs 8` 与 `--gpu-memory-utilization 0.90` 是**吞吐参数，刻意不复制**（benchmark 要用真实 serving 并发；greedy 解码内容不受 batching 影响的假设作为 limitation 记入 run manifest）。
- 共享常量放新模块 `scheduler_benchmark/contracts.py`（无循环依赖）：`MAX_ESTIMATED_TOKENS = 4096`、`RELIABLE = 1`、`UNRELIABLE = 0`。decision_service、gateway_transport、predictor 全部从这里 import，**不允许再出现第二个 4096 常量**。**Python transport 当前把 bool 写回 xargs（`gateway_transport.py:126`）——必须改成写 `RELIABLE`/`UNRELIABLE` int**；predictor 端严格解析（Python 里 `True == 1` 为真，所以必须 `isinstance(v, int) and not isinstance(v, bool)` 排除 bool）。
- decision 调用超时：**默认 2000 ms**（BertPredictor 固定 CPU，`predictor.py:57`；实测 warm 约 870–895 ms）。服务器上先实测 warm p99 再写入 run manifest；不许用未实测的 300 ms。
- score 语义：sigmoid(logit)，越大预测越长，调度越靠后（`train_ranker.py:140`）。
- rank→token 映射只用 `RankQuantileMapper`（`mapping_version="uncalibrated-rank-lookup-v1"`）；decision response 与 run manifest 都要携带 `mapping_version` + `approximation_notice` + quantile manifest sha256。
- 诚实红线：不伪造校准；confidence=0.9 是占位符；测试没跑就说没跑。
- 长任务一律 `nohup`，会话退出不得杀任务。
- Mac 上 pytest 走 `rtk proxy python3 -m pytest`；涉及 transformers/requests 的测试用 `.worktrees/final-training-artifacts/.venv/bin/python`。
- 服务器目标环境：恒源云镜像（CUDA/conda 预装、专属 `oss` CLI、交互式 `oss login`），不是通用 Ubuntu。

## 明确砍掉的范围（不是默默消失）

| 砍掉项 | 原因 | 去向 |
|---|---|---|
| gRPC sidecar OOD replay 线 | 不在 benchmark 主链路；sidecar schema 冲突留 PR 时解决 | 9 月 PR |
| confidence 真校准 | 无标定数据；占位已声明 | 论文 limitation |
| 两轮 tool-call E2E | benchmark 全单轮（first-invocation 截断是设计决定） | 不做 |
| 通用 Ubuntu bootstrap | 目标恒源云镜像 | 不做 |
| dashboard 接入（B8） | 依赖 benchmark 数据 | remaining-work B8 |

注意：v2 曾把 stock-vs-shim parity 砍掉——**v3 恢复**：`SCHEDULER_CLASS_TO_POLICY` 本来就含 `stock_fcfs`（stock vLLM Scheduler）与 `fcfs`（我们的 StockFCFSShim）两档（`runner.py:25-34`），矩阵各跑一遍即是 parity。网关开销对照用已有的成对重放 harness `scripts/run_gateway_overhead.py`（`gateway_overhead.py:54`），不再用语义错误的 `fcfs-direct` 档。

---

### Task 1: VeloxMesh fork 分支 — decision 注入 + 字段透传（Go）

**Files:**
- Modify: `/Users/alex/develop/VeloxMesh/internal/llm/types.go:88-99`
- Modify: `/Users/alex/develop/VeloxMesh/internal/http/handlers/chat.go:32-99`
- Create: `/Users/alex/develop/VeloxMesh/internal/ltr/decision.go`
- Create: `/Users/alex/develop/VeloxMesh/internal/ltr/decision_test.go`
- Create: `/Users/alex/develop/VeloxMesh/internal/providers/openai/adapter_ltr_test.go`
- Modify: `/Users/alex/develop/VeloxMesh/internal/gateway/service.go:180,454`
- Modify: `/Users/alex/develop/VeloxMesh/internal/providers/openai/adapter.go:106-127,203-225`

**Interfaces:**
- Consumes: decision service `POST /v1/decision`（Task 2 合同；`schema_version:"1.0"`）。
- Produces: 上游 body 注入 `vllm_xargs`/`chat_template_kwargs`/`stream_options`；env `LTR_DECISION_ENDPOINT`（空 → 全链 no-op）、`LTR_DECISION_TIMEOUT_MS`（默认 2000）。
- **信任边界：客户端 xargs 只白名单 `ltr_kind`、`ltr_category` 两个 key**；`ltr_tool_schema` 只用于 decision 请求、绝不进上游；其余客户端 key（包括伪造的 `workflow_estimated_tokens`/`prediction_reliable`/`decision_id`）一律丢弃。
- **响应验证（对齐 Python transport `gateway_transport.py:148` 的等价规则）**：`schema_version=="1.0"`、`decision_id == 期望值`、reliable→`estimated_tokens` 必须存在且 1..4096、unreliable→必须不存在；任何违约 fail-open。

已核实：`llm.Message.Content` 是 `string`（types.go:30）；两个调用点 `service.go:180/:454` 的变量名是 `req`；`openai.NewAdapter` 测试可构造（adapter.go:38）；上游 body 是 map 注入后无过滤（adapter.go:106/:203）；流式 usage 会转发（adapter.go:308→chat.go:187）；数据面单端口 `GATEWAY_DATA_ADDR`、`/healthz`（router.go:92）；auth 用 `DEV_API_KEY` bearer（auth.go:89-97）。

- [ ] **Step 1: fork + 分支**

```bash
gh repo fork zardonc/VeloxMesh --clone=false
cd /Users/alex/develop/VeloxMesh
git remote add fork "git@github.com:$(gh api user -q .login)/VeloxMesh.git" 2>/dev/null || true
git checkout -b feat/ltr-decision-adapter 4b4b5ad9d34fdeb8b87f21981338498f25035d09
```

- [ ] **Step 2: LLMRequest 扩字段**（types.go，`ToolChoice any` 后）

```go
	ClientXargs        map[string]any // client vllm_xargs; only ltr_* keys are trusted
	VLLMXargs          map[string]any // decision verdict, injected upstream
	ChatTemplateKwargs map[string]any
	StreamIncludeUsage bool
```

- [ ] **Step 3: proxyReq 透传**（chat.go；proxyReq 增 3 字段 + llmReq 赋值，与 v2 相同，此处 import 无变化）

```go
		VLLMXargs          map[string]any `json:"vllm_xargs,omitempty"`
		ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
		StreamOptions      *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options,omitempty"`
```

```go
		ClientXargs:        pReq.VLLMXargs,
		ChatTemplateKwargs: pReq.ChatTemplateKwargs,
		StreamIncludeUsage: pReq.StreamOptions != nil && pReq.StreamOptions.IncludeUsage,
```

- [ ] **Step 4: 失败测试**（`internal/ltr/decision_test.go`；注意 JSON 数字在 `map[string]any` 里 decode 成 float64，但我们写入的是 Go int——断言按写入类型）

```go
package ltr

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"veloxmesh/internal/llm"
)

func fakeDecision(t *testing.T, body map[string]any) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("bad decision request: %v", err)
		}
		for _, k := range []string{"schema_version", "request_id", "decision_id", "model_id", "request_age_ms", "messages", "generation_controls"} {
			if _, ok := req[k]; !ok {
				t.Errorf("decision request missing %q", k)
			}
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(body)
	}))
}

func testReq() *llm.LLMRequest {
	return &llm.LLMRequest{
		RequestID: "req-1",
		Model:     "qwen",
		Messages:  []llm.Message{{Role: "user", Content: "hello"}},
		ClientXargs: map[string]any{
			"ltr_kind": "tool", "ltr_category": "id:toolace",
			"ltr_tool_schema": "[{\"type\":\"function\"}]",
			// 恶意/陈旧字段，必须被丢弃：
			"workflow_estimated_tokens": 9999,
			"prediction_reliable":       1,
			"decision_id":               "spoofed",
		},
	}
}

func reliableBody() map[string]any {
	return map[string]any{
		"schema_version": "1.0", "decision_id": "dec-req-1",
		"prediction_reliable": true, "estimated_tokens": 512,
		"reliability_probability": 0.9, "ood_score": 0.0,
		"predictor_revision": "r", "feature_variant": "prompt_schema", "reason_code": "prediction_reliable",
	}
}

func TestReliableDecisionInjectsXargs(t *testing.T) {
	srv := fakeDecision(t, reliableBody())
	defer srv.Close()
	req := testReq()
	Apply(req, srv.URL, 2000)
	x := req.VLLMXargs
	if got, ok := x["workflow_estimated_tokens"].(int); !ok || got != 512 {
		t.Fatalf("workflow_estimated_tokens = %v", x["workflow_estimated_tokens"])
	}
	if x["prediction_reliable"] != 1 {
		t.Fatalf("prediction_reliable must be int 1, got %v", x["prediction_reliable"])
	}
	if x["decision_id"] != "dec-req-1" {
		t.Fatalf("decision_id = %v", x["decision_id"])
	}
	if _, leaked := x["ltr_tool_schema"]; leaked {
		t.Fatal("ltr_tool_schema must not reach upstream")
	}
	if x["ltr_kind"] != "tool" || x["ltr_category"] != "id:toolace" {
		t.Fatal("whitelisted client keys must be merged")
	}
}

func TestSpoofedClientFieldsAreDropped(t *testing.T) {
	req := testReq()
	Apply(req, "", 2000) // no endpoint → no-op, nothing injected at all
	if req.VLLMXargs != nil {
		t.Fatal("no endpoint must be a no-op")
	}
	srv := fakeDecision(t, map[string]any{
		"schema_version": "1.0", "decision_id": "dec-req-1",
		"prediction_reliable":     false,
		"reliability_probability": 0.1, "ood_score": 1.0,
		"predictor_revision": "r", "feature_variant": "prompt_schema", "reason_code": "ood_rejected",
	})
	defer srv.Close()
	req = testReq()
	Apply(req, srv.URL, 2000)
	if _, ok := req.VLLMXargs["workflow_estimated_tokens"]; ok {
		t.Fatal("spoofed estimate must not survive an unreliable verdict")
	}
	if req.VLLMXargs["prediction_reliable"] != 0 {
		t.Fatal("unreliable must be int 0")
	}
	if req.VLLMXargs["decision_id"] == "spoofed" {
		t.Fatal("spoofed decision_id must be dropped")
	}
}

func TestContractViolationsFailOpen(t *testing.T) {
	cases := []map[string]any{
		{"schema_version": "9.9", "decision_id": "dec-req-1", "prediction_reliable": true, "estimated_tokens": 512},
		{"schema_version": "1.0", "decision_id": "WRONG", "prediction_reliable": true, "estimated_tokens": 512},
		{"schema_version": "1.0", "decision_id": "dec-req-1", "prediction_reliable": true},                            // reliable 无 estimate
		{"schema_version": "1.0", "decision_id": "dec-req-1", "prediction_reliable": true, "estimated_tokens": 5000},  // 超 4096
		{"schema_version": "1.0", "decision_id": "dec-req-1", "prediction_reliable": false, "estimated_tokens": 512}, // unreliable 带 estimate
	}
	for i, body := range cases {
		srv := fakeDecision(t, body)
		req := testReq()
		Apply(req, srv.URL, 2000)
		srv.Close()
		if _, ok := req.VLLMXargs["workflow_estimated_tokens"]; ok {
			t.Fatalf("case %d: contract violation must fail open (no estimate)", i)
		}
		if req.VLLMXargs["prediction_reliable"] != 0 {
			t.Fatalf("case %d: fail-open must mark unreliable", i)
		}
	}
}

func TestDecisionFailureFailsOpen(t *testing.T) {
	req := testReq()
	Apply(req, "http://127.0.0.1:1", 100)
	if req.VLLMXargs["prediction_reliable"] != 0 {
		t.Fatal("fail-open must mark prediction_reliable=0")
	}
	if _, ok := req.VLLMXargs["workflow_estimated_tokens"]; ok {
		t.Fatal("fail-open must not invent tokens")
	}
}
```

- [ ] **Step 5: 确认编译失败** `GOTOOLCHAIN=auto go test ./internal/ltr/`

- [ ] **Step 6: 实现 `internal/ltr/decision.go`**

```go
// Package ltr calls the external LTR decision service and stamps the
// verdict onto the request as vllm_xargs for the upstream engine.
// Contract: vllm_xargs values must be str/int/float/list (no bool) —
// prediction_reliable is int 0/1.
package ltr

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"veloxmesh/internal/llm"
)

const (
	schemaVersion     = "1.0"
	maxEstimatedTokens = 4096
)

var generationControls = map[string]any{
	"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 4096,
}

// clientWhitelist is the only client xargs allowed upstream.
var clientWhitelist = map[string]bool{"ltr_kind": true, "ltr_category": true}

// EnvIntDefault reads an int env var with a fallback.
func EnvIntDefault(key string, def int) int {
	if raw := os.Getenv(key); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil {
			return v
		}
	}
	return def
}

// Apply calls the decision service and sets req.VLLMXargs.
// endpoint=="" → no-op. Any failure or contract violation fails open:
// prediction_reliable=0 and no token estimate.
func Apply(req *llm.LLMRequest, endpoint string, timeoutMs int) {
	if endpoint == "" {
		return
	}
	decisionID := "dec-" + req.RequestID
	base := map[string]any{
		"prediction_reliable": 0,
		"decision_id":         decisionID,
		"workflow_id":         req.RequestID,
		"step_id":             "0",
	}
	for k, v := range req.ClientXargs {
		if clientWhitelist[k] {
			base[k] = v
		}
	}
	req.VLLMXargs = base

	messages := make([]map[string]any, 0, len(req.Messages))
	for _, m := range req.Messages {
		messages = append(messages, map[string]any{"role": string(m.Role), "content": m.Content})
	}
	payload := map[string]any{
		"schema_version":       schemaVersion,
		"request_id":           req.RequestID,
		"decision_id":          decisionID,
		"model_id":             req.Model,
		"request_age_ms":       0,
		"messages":             messages,
		"generation_controls":  generationControls,
		"workflow_id":          req.RequestID,
		"step_id":              "0",
		"conversation_id":      req.RequestID,
		"previous_tool_gap_ms": 0,
	}
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}
	if ts, ok := req.ClientXargs["ltr_tool_schema"].(string); ok && ts != "" {
		payload["tool_schema_text"] = ts
	}
	body, err := json.Marshal(payload)
	if err != nil {
		slog.Warn("ltr decision marshal failed; fail-open", "err", err)
		return
	}
	client := &http.Client{Timeout: time.Duration(timeoutMs) * time.Millisecond}
	resp, err := client.Post(endpoint+"/v1/decision", "application/json", bytes.NewReader(body))
	if err != nil {
		slog.Warn("ltr decision call failed; fail-open", "err", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		slog.Warn("ltr decision non-200; fail-open", "status", resp.StatusCode)
		return
	}
	var verdict struct {
		SchemaVersion      string `json:"schema_version"`
		DecisionID         string `json:"decision_id"`
		PredictionReliable bool   `json:"prediction_reliable"`
		EstimatedTokens    *int   `json:"estimated_tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&verdict); err != nil {
		slog.Warn("ltr decision decode failed; fail-open", "err", err)
		return
	}
	// Contract validation — mirror of Python gateway_transport rules.
	switch {
	case verdict.SchemaVersion != schemaVersion,
		verdict.DecisionID != decisionID,
		verdict.PredictionReliable && (verdict.EstimatedTokens == nil ||
			*verdict.EstimatedTokens < 1 || *verdict.EstimatedTokens > maxEstimatedTokens),
		!verdict.PredictionReliable && verdict.EstimatedTokens != nil:
		slog.Warn("ltr decision contract violation; fail-open",
			"schema", verdict.SchemaVersion, "decision_id", verdict.DecisionID)
		return
	}
	if verdict.PredictionReliable {
		req.VLLMXargs["prediction_reliable"] = 1
		req.VLLMXargs["workflow_estimated_tokens"] = *verdict.EstimatedTokens
	}
}
```

- [ ] **Step 7: 测试至绿** `GOTOOLCHAIN=auto go test ./internal/ltr/` — 期望 5 PASS。

- [ ] **Step 8: 调用点**（service.go `:180`/`:454` `ProcessRequest` 之后各一行；import 补 `"os"`、`"veloxmesh/internal/ltr"`）

```go
	ltr.Apply(req, os.Getenv("LTR_DECISION_ENDPOINT"), ltr.EnvIntDefault("LTR_DECISION_TIMEOUT_MS", 2000))
```

- [ ] **Step 9: adapter 注入**（Complete 与 Stream 的 `json.Marshal` 前各一份；Stream 另加 stream_options）

```go
	if req.VLLMXargs != nil {
		openAIReq["vllm_xargs"] = req.VLLMXargs
	}
	if req.ChatTemplateKwargs != nil {
		openAIReq["chat_template_kwargs"] = req.ChatTemplateKwargs
	}
```

```go
	if req.StreamIncludeUsage {
		openAIReq["stream_options"] = map[string]any{"include_usage": true}
	}
```

- [ ] **Step 10: adapter 端到端测试**（`adapter_ltr_test.go`）：httptest 假 upstream 捕获 body → `openai.NewAdapter`（构造见 adapter.go:38）指向它 → 带三字段的 LLMRequest 走 `Complete` 与 `Stream` → 断言捕获 body 含 `vllm_xargs`（其中 `prediction_reliable` 为数字 1 而非 bool）、`chat_template_kwargs`、`stream_options`。

- [ ] **Step 11: 构建 + vet + 全测**

```bash
GOTOOLCHAIN=auto go build ./... && GOTOOLCHAIN=auto go vet ./... && GOTOOLCHAIN=auto go test ./internal/...
GOTOOLCHAIN=auto go build -o bin/gateway ./cmd/gateway
```

- [ ] **Step 12: commit + push + 记录 SHA**

```bash
git add internal/llm/types.go internal/http/handlers/chat.go internal/ltr/ internal/gateway/service.go internal/providers/openai/
git commit -m "feat(gateway): LTR decision-service adapter — inject vllm_xargs + passthrough chat_template_kwargs/stream_options"
git push -u fork feat/ltr-decision-adapter
git rev-parse HEAD   # → 写入主仓 scripts/server/manifest/gateway-pin.txt（Task 6 消费）
```

---

### Task 2: decision service — contracts 模块、tool_schema_text、mapper、4096

**Files:**
- Create: `scheduler_benchmark/contracts.py`
- Modify: `scheduler_benchmark/decision_service.py`、`scheduler_benchmark/gateway_transport.py:188`、`scripts/run_decision_service.py`
- Test: `tests/test_decision_service.py`（真实 helper：`valid_request` :15、`make_app` :48）、`tests/test_gateway_transport.py`

**Interfaces:**
- `contracts.py`：`MAX_ESTIMATED_TOKENS = 4096`。
- `/v1/decision` 请求新增可选 `tool_schema_text`（str，1..262144 字节，优先于 system-message 提取）。
- `generation_controls` 支持档 `max_tokens: 4096`（**同步改 `valid_request` fixture 的 2048→4096**）。
- `DecisionApplication.__init__` 增 `quantile_mapper=None, quantile_manifest_sha256=None` 两参数（`make_app` 透传）；mapper 非 None 时 `estimated_tokens = max(1, min(4096, round(mapper.map_score(score).quantiles[50])))`，且 **reliable 与 unreliable 两种 response 都**携带 `"mapping_version"`、`"approximation_notice"`、`"quantile_manifest_sha256"` 三字段（provenance 不因判决而消失）；mapper 为 None 保留旧线性式且不带三字段（既有单测兼容——**但既有精确匹配测试 `test_reliable_prediction_echoes_decision_id_and_estimate` 期望 512 = 0.25×2048，改 4096 后线性式变 1024，期望值必须同步更新**）。
- **`quantile_manifest_sha256` 的定义 = manifest 文件原始 bytes 的 SHA-256**，由 CLI 装载时计算传入；直接用内存 dict 构造 mapper 的测试传显式假值（如 `"test-sha"`）。
- **Python transport 同步改合同**（`gateway_transport.py`）：`prediction_reliable` 写 `RELIABLE`/`UNRELIABLE` int（当前 :126 是 bool）；`_validate_bundle`（:148-153）required fields 增加三个 provenance 字段，且校验精确值：`mapping_version == "uncalibrated-rank-lookup-v1"`、`approximation_notice == APPROXIMATION_NOTICE`、sha256 为 64 位 hex。**transport/生产路径只接受 mapper-backed 响应**（三字段必须存在）；`quantile_mapper=None` 只是旧单测的兼容形态，不进生产链。反例测试：transport 输出不得含 bool 值。
- CLI `--quantile-manifest` required；manifest 缺失/损坏 → stderr 原因 + exit 2。
- **manifest fixture 显式定义**（新 helper，放 `tests/test_decision_service.py`；`_validate_manifest` 要求 0..100 全 percentile + global 50/70/90 + 正确 mapping_version/approximation_notice）：

```python
def minimal_quantile_manifest() -> dict[str, object]:
    from scheduler_benchmark.rank_quantiles import APPROXIMATION_NOTICE, MAPPING_VERSION

    return {
        "mapping_version": MAPPING_VERSION,
        "model_version": "test-model",
        "approximation_notice": APPROXIMATION_NOTICE,
        "sample_count": 6000,
        "percentiles": {str(p): float(10 + 5 * p) for p in range(10, 100)},
        "global_quantiles": {"50": 260.0, "70": 360.0, "90": 460.0},
    }
    # 真实合同：percentile 范围是 p10..p99（rank_quantiles.py:19 MIN/MAX_PERCENTILE），
    # 且必须带 sample_count==6000（rank_quantiles.py:196）。
```

- [ ] **Step 1: 失败测试**（用真实 helper 名）

```python
def test_tool_schema_text_field_takes_precedence() -> None:
    request = valid_request()
    request["tool_schema_text"] = "RAW-SCHEMA"
    request["messages"] = [
        {"role": "system", "content": "SYSTEM-SCHEMA"},
        {"role": "user", "content": "hi"},
    ]
    predictor_input = decision_service._predictor_input(
        decision_service._validate_request(request)
    )
    assert predictor_input.metadata["tool_schema_text"] == "RAW-SCHEMA"


def test_estimated_tokens_use_quantile_mapper() -> None:
    mapper = RankQuantileMapper(minimal_quantile_manifest())
    app = make_app(quantile_mapper=mapper, quantile_manifest_sha256="test-sha")
    response = app.decide(valid_request())
    expected = max(1, min(4096, round(mapper.map_score(0.25).quantiles[50])))
    assert response["estimated_tokens"] == expected
    assert response["mapping_version"] == "uncalibrated-rank-lookup-v1"
    assert "approximation_notice" in response
    assert response["quantile_manifest_sha256"] == "test-sha"


def test_unreliable_response_still_carries_provenance() -> None:
    mapper = RankQuantileMapper(minimal_quantile_manifest())
    app = make_app(ood=True, quantile_mapper=mapper, quantile_manifest_sha256="test-sha")
    response = app.decide(valid_request())
    assert response["prediction_reliable"] is False
    assert "estimated_tokens" not in response
    assert response["mapping_version"] == "uncalibrated-rank-lookup-v1"
    assert response["quantile_manifest_sha256"] == "test-sha"


def test_generation_controls_accept_max_tokens_4096() -> None:
    request = valid_request()
    request["generation_controls"]["max_tokens"] = 4096
    decision_service._validate_request(request)  # 不抛
```

- [ ] **Step 2: 确认失败** `rtk proxy python3 -m pytest tests/test_decision_service.py -x -q`
- [ ] **Step 3: 实现**：`contracts.py`（3 行常量模块）；decision_service/gateway_transport 改 import；`valid_request` fixture max_tokens 4096；`test_reliable_prediction_echoes_decision_id_and_estimate` 期望 512→1024 并补 `reason_code` 键（若响应结构对比是全量 dict）；`_validate_request` 收 `tool_schema_text`；`_predictor_input` 优先用它；`DecisionApplication.__init__` 增 `quantile_mapper=None`（`make_app` 透传）；`decide` mapper 分支 + 三个新 response 字段（`quantile_manifest_sha256` 在 CLI 装载时算好传入）；CLI `--quantile-manifest` required + exit 2。
- [ ] **Step 4: 全绿** `rtk proxy python3 -m pytest tests/test_decision_service.py tests/test_gateway_transport.py tests/test_decision_service_cli.py tests/test_mock_stack.py -q`
- [ ] **Step 5: commit** `git commit -m "feat(decision): contracts module, tool_schema_text passthrough, quantile mapper, 4096 cap"`

---

### Task 3: 调度器 gateway predictor + 顺序审计日志

**Files:**
- Modify: `scheduler_benchmark/predictor.py`（新增 `GatewayMetadataPredictor`）
- Modify: `scheduler_benchmark/vllm_scheduler.py`（`build_predictor_from_env`、`reorder_request_queue`）
- Test: `tests/test_vllm_scheduler.py`（真实 fixture：`FakeRequest(request_id, arrival_time, prompt_token_ids, trace_headers, sampling_params=None)` dataclass :20、`FakeQueue` :29）

**Interfaces:**
- `LTR_PREDICTOR=gateway`；metadata `prediction_reliable` **是 int（排除 bool——Python `True == 1` 为真，必须 isinstance 检查）且 == RELIABLE**、`workflow_estimated_tokens` 为 int ≥1 → `Prediction(score=min(est,4096)/4096, confidence=0.9, ood=False)`；否则（含 bool `True`）→ `Prediction(1.0, 0.0, True)`。常量引用 `contracts`，不新设。
- `LTR_ORDER_LOG` 设置时**每次调用都写**（含 len<2 的 early-return 分支——当前 `vllm_scheduler.py:101` 在 <2 时直接 return，单请求会拿不到证据），行含 per-request 审计：`{"policy":..., "order":[...], "predictions": {request_id: {"score":..., "ood":...}}}`。
- **import 事实**：`vllm_scheduler.py:5` 当前**没有** import `json`（`os` 有）——实现时补 `import json`。测试文件需补 `from types import SimpleNamespace`、`import json`（若尚未有）。

- [ ] **Step 1: 失败测试**

```python
def test_gateway_predictor_consumes_injected_estimate(monkeypatch) -> None:
    monkeypatch.setenv("LTR_PREDICTOR", "gateway")
    predictor = build_predictor_from_env()
    request = FakeRequest("r1", 0.0, [], {})
    request.sampling_params = SimpleNamespace(
        extra_args={
            "workflow_estimated_tokens": 1024,
            "prediction_reliable": 1,   # int 合同，vLLM 协议无 bool
            "ltr_kind": "tool",
        }
    )
    metadata = _request_metadata(request)
    prediction = predictor.predict(
        PredictorInput(request_id="r1", prompt_token_ids=(), metadata=metadata)
    )
    assert prediction.score == pytest.approx(1024 / 4096)
    assert prediction.confidence == pytest.approx(0.9)
    assert prediction.ood is False


def test_gateway_predictor_falls_back_when_absent_or_unreliable(monkeypatch) -> None:
    monkeypatch.setenv("LTR_PREDICTOR", "gateway")
    predictor = build_predictor_from_env()
    for metadata in ({}, {"prediction_reliable": 0}, {"prediction_reliable": 1}):
        prediction = predictor.predict(
            PredictorInput(request_id="r2", prompt_token_ids=(), metadata=metadata)
        )
        assert (prediction.score, prediction.confidence, prediction.ood) == (1.0, 0.0, True)


def test_gateway_predictor_rejects_bool_true_as_contract_violation(monkeypatch) -> None:
    # Python True == 1；如果 transport 违约输出 bool，predictor 必须回退而不是掩盖
    monkeypatch.setenv("LTR_PREDICTOR", "gateway")
    predictor = build_predictor_from_env()
    prediction = predictor.predict(
        PredictorInput(
            request_id="r3",
            prompt_token_ids=(),
            metadata={"prediction_reliable": True, "workflow_estimated_tokens": 1024},
        )
    )
    assert (prediction.score, prediction.confidence, prediction.ood) == (1.0, 0.0, True)


def test_order_log_records_reorder(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LTR_ORDER_LOG", str(tmp_path / "order.jsonl"))
    queue = FakeQueue(
        [
            FakeRequest("long", 1.0, [1, 2, 3], {}),
            FakeRequest("short", 2.0, [1], {}),
        ]
    )
    predictions = {
        request.request_id: Prediction(1.0, 0.0, True, 0.0) for request in queue
    }
    reorder_request_queue(queue, "prompt_sjf", predictions, now_s=3.0)
    rows = [json.loads(line) for line in (tmp_path / "order.jsonl").read_text().splitlines()]
    assert rows[-1]["policy"] == "prompt_sjf"
    assert rows[-1]["order"] == ["short", "long"]
    assert set(rows[-1]["predictions"]) == {"short", "long"}


def test_order_log_written_even_for_single_request(tmp_path, monkeypatch) -> None:
    # len<2 会 early-return，但审计日志必须仍然落盘（租卡日单请求门槛的证据链）
    monkeypatch.setenv("LTR_ORDER_LOG", str(tmp_path / "order.jsonl"))
    queue = FakeQueue([FakeRequest("only", 1.0, [1], {})])
    predictions = {"only": Prediction(0.5, 0.9, False, 0.0)}
    reorder_request_queue(queue, "pure_ltr", predictions, now_s=3.0)
    rows = [json.loads(line) for line in (tmp_path / "order.jsonl").read_text().splitlines()]
    assert rows[-1]["order"] == ["only"]
    assert rows[-1]["predictions"]["only"]["score"] == 0.5
```

- [ ] **Step 2: 确认失败** `rtk proxy python3 -m pytest tests/test_vllm_scheduler.py -x -q`
- [ ] **Step 3: 实现**（predictor.py）：

```python
from scheduler_benchmark.contracts import MAX_ESTIMATED_TOKENS, RELIABLE


class GatewayMetadataPredictor:
    """Consumes gateway-injected estimates from request metadata.

    Contract: prediction_reliable is int 0/1 (vLLM extra_args has no bool);
    score = min(est, 4096)/4096 — a declared monotone mapping, not a
    calibration.
    """

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        metadata = predictor_input.metadata or {}
        flag = metadata.get("prediction_reliable")
        reliable = (
            isinstance(flag, int) and not isinstance(flag, bool) and flag == RELIABLE
        )
        raw = metadata.get("workflow_estimated_tokens")
        if reliable and isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
            score = min(raw, MAX_ESTIMATED_TOKENS) / MAX_ESTIMATED_TOKENS
            return Prediction(score=score, confidence=0.9, ood=False, latency_ms=0.0)
        return Prediction(score=1.0, confidence=0.0, ood=True, latency_ms=0.0)
```

`build_predictor_from_env` 加 `gateway` 分支。`reorder_request_queue` 的 order-log 改为**函数入口统一写**（early-return 前）：

```python
def _write_order_log(requests, predictions, policy) -> None:
    log_path = os.environ.get("LTR_ORDER_LOG")
    if not log_path:
        return
    entry = {
        "policy": str(policy),
        "order": [request.request_id for request in requests],
        "predictions": {
            request.request_id: {
                "score": predictions[request.request_id].score,
                "ood": predictions[request.request_id].ood,
            }
            for request in requests
            if request.request_id in predictions
        },
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
```

`reorder_request_queue`：len<2 时也调用 `_write_order_log(requests, predictions, policy)` 后再 return；重排后用 ordered 顺序调用。**补 `import json`（vllm_scheduler.py:5 目前没有）**。order-log 测试期望值相应带 `predictions` 键。
- [ ] **Step 4: 全绿** `rtk proxy python3 -m pytest tests/test_vllm_scheduler.py tests/test_predictor.py -q`
- [ ] **Step 5: commit** `git commit -m "feat(scheduler): gateway metadata predictor (int-contract) + reorder audit log"`

---

### Task 4: tier2 转正 + 依赖拆分 + OOD wrapped-tools 兼容

**Files:**
- Create: `ltr_training/tier2.py`、`scripts/replay_tier2_labels.py`（worktree 逐字复制）
- Create: `requirements/train.txt`（Mac/训练侧，来自 worktree freeze 实测）与 `requirements/server.in`（GPU serving 侧输入集）
- Test: `tests/test_tier2_contract.py`（新）

**Interfaces:**
- `build_request(row, *, model, max_tokens=4096)`（keyword-only `model`）。
- **依赖拆分（v2 的单一 requirements.txt 是 pin 冲突：worktree `torch==2.13.0`，而 vllm 0.24.0 要求 `torch==2.11.0`）**：`requirements/train.txt` = worktree freeze 实测（含 torch 2.13、transformers、requests）；`requirements/server.in` = `vllm==0.24.0`、`aiohttp`、`requests`、`transformers`（**不 pin torch**，由 vllm 依赖解析出 2.11 组合）；服务器装完后 `pip freeze > /hy-tmp/ltr/manifest.pip.txt` 即为 serving lock 证据（Task 6）。
- **OOD schema 是 OpenAI wrapped 格式**：`ood_conversion.py:143` 产出 `canonical_schema(tools)`，tools 为 `{"type":"function","function":{...}}`（`test_ood_conversion.py:94`）。当前 `_toolace_tools`（tier2.py:31）只认 ToolACE 扁平格式 → **允许最小扩展：识别"已是 wrapped tools"的 JSON 数组时原样返回**（不重命名、不再包一层）。否则 OOD 请求会退化成 system message，decision service 因缺 `tools` 判 missing optional features。

- [ ] **Step 1: 复制两文件（verbatim）+ 生成两份依赖文件（train.txt 来自 `.worktrees/final-training-artifacts/.venv/bin/pip freeze` 实测；server.in 手写 4 行）**
- [ ] **Step 2: 失败测试**

```python
import json

from ltr_training.tier2 import build_request, _toolace_tools


def test_build_request_matches_label_generation_shape_with_history() -> None:
    row = {
        "prompt": "final question",
        "tool_schema": "",
        "history": [["human", "earlier turn"], ["gpt", "earlier answer"]],
    }
    request = build_request(row, model="qwen")
    assert request["messages"] == [
        {"role": "user", "content": "earlier turn"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "final question"},
    ]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["max_tokens"] == 4096
    assert request["temperature"] == 0


def test_toolace_tools_passes_through_wrapped_openai_tools() -> None:
    wrapped = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    tools = _toolace_tools(wrapped)
    assert tools and tools[0]["function"]["name"] == "get_weather"
```

- [ ] **Step 3: 实现**：wrapped 测试若失败 → `_toolace_tools` 开头加"元素含 `type=="function"` 且有 `function.name`"的识别分支，原样返回；独立 commit。
- [ ] **Step 4: 验证** `py_compile` 两文件 + worktree venv 跑 `tests/test_tier2_contract.py`。
- [ ] **Step 5: commit** `git commit -m "feat(training): promote tier2 replay, split train/server deps, wrapped-tools passthrough"`

---

### Task 5: runner chat 迁移 + workload v2（含 history）+ mock stack 同步

**Files:**
- Modify: `scheduler_benchmark/runner.py`（`make_completion_payload`→`make_chat_payload`、`stream_completion`、`load_workload`、`WorkloadRequest`）
- Modify: `ltr_training/workload_builder.py:88-108`
- Modify: `scheduler_benchmark/mock_stack.py`
- Test: `tests/test_runner.py`、`tests/test_workload_builder.py`、`tests/test_mock_stack.py`

**Interfaces:**
- workload JSONL v2：`prompt` = 原始 user 文本、`tool_schema`（str，必填）、**`history`（list[[role, text]]，必填，可为空表）**；manifest `schema_version: "offline-workload-v2"`；缺任一字段 → ValueError。
- `WorkloadRequest` 增 `tool_schema: str`、`history: list`。
- `make_chat_payload(request, *, model)` = `build_request({"prompt": request.prompt, "tool_schema": request.tool_schema, "history": request.history}, model=model)` 合并 `{"stream": True, "stream_options": {"include_usage": True}, "vllm_xargs": {"ltr_kind": ..., "ltr_category": ..., "ltr_tool_schema": request.tool_schema}}`。
- SSE 解析：`choices[0].delta` 的 `content` 或 `tool_calls` 任一非空计首 token；usage 捕获不变。

- [ ] **Step 1: 失败测试**

```python
def test_make_chat_payload_matches_tier2_shape() -> None:
    request = WorkloadRequest(
        request_id="r1", prompt="final", baseline_service_ms=10.0,
        max_tokens=4096, kind="tool", category="id:toolace",
        tool_schema="[]", history=[["human", "prior"]],
    )
    payload = make_chat_payload(request, model="qwen")
    assert payload["messages"][0] == {"role": "user", "content": "prior"}
    assert payload["messages"][-1] == {"role": "user", "content": "final"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["vllm_xargs"]["ltr_tool_schema"] == "[]"
    assert payload["vllm_xargs"]["ltr_category"] == "id:toolace"
```

加 chat SSE fixture 测试（手写 `delta.content`/`delta.tool_calls`/usage/[DONE] 行）与 workload v2 严格加载测试（缺 `history` → ValueError）。

- [ ] **Step 2: 实现**（loader 校验、payload、SSE、workload_builder 传 `item.history`、mock_stack chat SSE 化）。
- [ ] **Step 3: 全绿** `rtk proxy python3 -m pytest tests/test_runner.py tests/test_workload_builder.py tests/test_mock_stack.py -q`
- [ ] **Step 4: 重建 workload v2**：`scripts/build_offline_workload.py` 出 v2；抽查 3 行：`prompt` 无 `[USER]` 前缀、ToolACE 行 `tool_schema` 非空、含 `history` 键。
- [ ] **Step 5: commit** `git commit -m "feat(runner): chat migration with history — tier2-consistent shape, workload v2"`

---

### Task 6: 服务器脚本（scripts/server/，恒源云）+ OSS 重打包

**Files:**
- Create: `scripts/server/inventory_and_repack.sh`（Mac 一次性）、`manifest/oss-objects.json`、`manifest/gateway-pin.txt`
- Create: `restore_from_oss.sh`、`setup_env.sh`、`build_gateway.sh`、`launch_vllm.sh`、`launch_decision.sh`、`launch_gateway.sh`、`calibrate_saturation.sh`、`run_matrix.sh`
- Modify: `scripts/server/README.md`

**真实起点（已实测 T7）：** `/Volumes/T7 Shield/vllm-ltr-results/` 只有 `tier2-checkpoints.tar`（5857.7M，成员布局是 `tier2-matrix/.../final/`，**不是** `checkpoints_best_predictor*`）和 `tier2-results.tar.gz`（5077.9M）；v2 写的 `tier2-results-0718-0412.tar.gz`/`lmcache-labels.tar.gz` 不存在。

**trap 规则修正（v2 会杀掉刚起的服务）：** 独立 launcher 只 `trap INT TERM`，health 通过后 `trap - EXIT` 再正常退出；**只有** `run_matrix.sh` 和 `smoke_local_chain.sh` 这类编排器持有 `trap cleanup EXIT INT TERM`。

**scheduler 参数修正：** `SCHEDULER_CLASS_TO_POLICY`（runner.py:25-34）key 是 FQCN：vLLM 与 runner **都传同一 FQCN**（如 `--scheduler-cls scheduler_benchmark.vllm_scheduler.PureLTRScheduler`；vLLM 侧同为点分路径，不用冒号）。矩阵循环 = 该 dict 除 `stock_fcfs` 外 6 个 key；另加一档 `stock_fcfs`（`vllm.v1.core.sched.scheduler.Scheduler`）作 stock-vs-shim parity。

**quantile 输入的真实状况（v3 复核实测）：** `tier2-results.tar.gz` 里唯一 JSONL 是 `tier2-toolace-6000-ledger.jsonl`——6000 行、ok=5997、error=3、**没有 prompt/tool_schema 字段**（ledger 写入只存 sample/source/output_length，`tier2.py:149`）。prompt/tool_schema 在**输入侧文件** `tier2-toolace-sample-6000.jsonl`（训练 manifest 记录其 SHA `ee5a5889...`、原路径 `/hy-tmp/results/`）。所以合并输入必须两文件 join，且 3 行 error 需要租卡日补 replay。

- [ ] **Step 1: `inventory_and_repack.sh`（Mac 跑一次）**：
  1. `tar tf tier2-checkpoints.tar` 记录成员路径（三个 seed 在 `tier2-matrix/.../bert-prompt_schema-tier2-seed{17,42,73}/final/`，已核实存在）；抽取重命名为 `checkpoints_best_predictor{,_seed42,_seed73}`；
  2. **重建 `tier2-toolace-sample-6000.jsonl`（已实测：文件本体不在 T7 任何归档里，只有 manifest）**。重建配方（`tier2-sample-manifest.json` 全记录，已核实）：源 = tier1 文件 `toolace-6bda777-qwen35.jsonl`（SHA `6dc808aa8f76a5391d33c22ecb0ae2a2967d01c923c71ec85d84ec537e5f227b`，本地无 → 租卡日从 `oss://lmcache-labels.tar.gz` 恢复后定位；不在则从 pinned ToolACE 6bda777 重建 tier1）+ `sampling_seed=42` + 原采样脚本 → 输出 SHA 必须 = `ee5a5889ca3d9bbee7790e7a408bd1664a285b6410b4fee54e45786d3eecb709`（manifest 里还有全量 `sample_ids` 可逐 ID 核对）；不一致 → 脚本失败，不许静默继续。此步移到**租卡日 restore 之后、quantile 构建之前**执行；
  3. 校验 ledger：6000 个唯一 `sample_id`、ok=5997/error=3 如实写进 repack 清单（**不要求原始行唯一**——replay 是追加式，重放后同一 sample_id 会有多行，`tier2.py:183`）；
  4. 把 sample-6000、ledger、Task 5 workload v2 + manifests 打成 `benchmark-bundle.tar.gz`；
  5. `shasum -a 256` + `stat -f %z` 写 `manifest/oss-objects.json`（对象：`benchmark-bundle.tar.gz` + 既有 `tier2-checkpoints.tar`、`tier2-results.tar.gz`，含 `unpacks_to` 真实路径表）；**0/空串进 commit = 未完成**；
  6. `oss cp` 上传 + `oss ls` 读回校验。

- [ ] **Step 1b: `merge_quantile_labels.py`（新脚本，服务器上在补 replay 后运行）**：ledger 先按行序做 **sample_id latest-wins** 去重（replay 追加不覆盖，`tier2.py:85/:183`），再 join sample-6000（prompt/tool_schema）→ `labels-merged-6k.jsonl`（四字段 `sample_id/prompt/tool_schema/output_length`，`rank_quantiles.py:154`）；硬校验：恰好 6000 个唯一 sample_id、全部 `status=ok` 且有 output_length（**mapper manifest 硬编码 `sample_count==6000`，`rank_quantiles.py:196`；censored 行仍有长度，缺的是 3 行 error**）。**补 replay 后仍不满 6000 → NO-GO，停止矩阵**（不存在"传实际 expected-count"的降级——manifest 合同会在 decision service 装载时拒绝非 6000）。
- [ ] **Step 2: `restore_from_oss.sh`**：`oss ls` 预检 → 按 manifest 逐对象 `oss cp` 到 `.partial-` → sha256 校验 → `mv` 原子就位 → 解包 `/hy-tmp/ltr/artifacts/<sha8>/` → `ln -sfn` `artifacts/current` → 三 checkpoint 前向 smoke（PY 块同前）。**quantile 构建链（vLLM 起来后）**：
  1. 补 replay 3 行 error：`scripts/replay_tier2_labels.py`（既有 resume 语义只重放缺失行）；
  2. `python3 scripts/server/merge_quantile_labels.py --samples .../tier2-toolace-sample-6000.jsonl --ledger .../tier2-toolace-6000-ledger.jsonl --output .../labels-merged-6k.jsonl`；
  3. 构建：

```bash
python3 scripts/build_rank_quantiles.py \
  --labels /hy-tmp/ltr/artifacts/current/labels-merged-6k.jsonl \
  --checkpoint /hy-tmp/ltr/artifacts/current/checkpoints_best_predictor \
  --sidecar-output /hy-tmp/ltr/artifacts/current/replay-sidecar.jsonl \
  --manifest-output /hy-tmp/ltr/artifacts/current/rank_quantiles.json \
  --model-version bert-prompt_schema-tier2-seed17 --expected-count 6000
```

（`--expected-count` 固定 6000；merge 已保证不满 6000 即 NO-GO。）

- [ ] **Step 3: `setup_env.sh`**：python ≥3.10 断言 → venv → `pip install -r requirements/server.in` → `pip freeze > /hy-tmp/ltr/manifest.pip.txt`（serving lock 证据）→ vllm 0.24 断言 → `HF_ENDPOINT=https://hf-mirror.com HF_HOME=/hy-tmp/hf huggingface-cli download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir /hy-tmp/models/qwen3.5-9b` → `df /hy-tmp` ≥60G 断言。
- [ ] **Step 3b: decision 超时实测（`measure_decision_latency.sh`，在 decision service 起来之后、gateway 起来之前跑）**：走**真 HTTP `/v1/decision` 路径**、并发 8（≈矩阵档位的网关并发量级）；**先 20 次 warm-up 不计入统计**，再采集 200 个样本；`timeout_ms = max(2000, ceil(1.25 * concurrent_p99_ms))` 写 `/hy-tmp/ltr/manifest.decision-latency.json`（含原始 200 个样本）。10 次串行 p99 不够——没覆盖 HTTP、并发竞争，也没 margin。
- [ ] **Step 4: `build_gateway.sh`**：`git clone <fork-url> && git checkout $(cat scripts/server/manifest/gateway-pin.txt)` → go 官方 tarball（无 go 时）→ `GOTOOLCHAIN=auto go build -o /hy-tmp/ltr/bin/gateway ./cmd/gateway` → `go test ./internal/ltr/`。
- [ ] **Step 5: 三个 launch 脚本**（trap 规则见上；pidfile/health 断言同 v2）：
  - `launch_vllm.sh <FQCN> <RUN_TAG>`：`PYTHONPATH=/hy-tmp/ltr/repo LTR_PREDICTOR=gateway LTR_ORDER_LOG=/hy-tmp/ltr/runs/<RUN_TAG>/order.jsonl nohup python -m vllm.entrypoints.openai.api_server --model /hy-tmp/models/qwen3.5-9b --served-model-name qwen3.5-9b --dtype bfloat16 --port 8000 --scheduler-cls <FQCN> --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' --max-model-len 8192`（**后五个参数复刻 Tier-2 打标环境**，`run_post_matrix_tier2.sh:54`；缺 tool parser 会改变 tool-call 回应形状，true_length 对比失效）。
  - `launch_decision.sh`：`--predictor bert --checkpoint .../checkpoints_best_predictor --quantile-manifest .../rank_quantiles.json --port 9200`（flag 名以 `scripts/run_decision_service.py` 实际 CLI 为准，执行时核对）。
  - `launch_gateway.sh`：env 同 v2 + `LTR_DECISION_TIMEOUT_MS=$(jq .timeout_ms /hy-tmp/ltr/manifest.decision-latency.json)`（来自 Step 3b 的实测公式，下限 2000）。
- [ ] **Step 6: `calibrate_saturation.sh`**：120 请求子集；网格 `0.3 0.45 0.68 1.0 1.5 2.2`；每档 runner `--scenario steady --load 90 --repeats 1 --capacity-rps $rps`；achieved_rps 从 runner 输出（`summarize_samples` 已含 completed/throughput，runner.py:263）；**失败规则**：第一档即 `achieved/(0.9*rps)<0.95` → exit 1（网格下限太高）；全网格都 ≥0.95 → exit 1（上限太低）；否则取饱和前一档写 `capacity.json`（全网格数据保留）。
- [ ] **Step 7: `run_matrix.sh`**：
  - mixed@90 主矩阵：6 FQCN（`SCHEDULER_CLASS_TO_POLICY` 除 stock）× `--repeats 3` + `stock_fcfs` parity 档 = 7 档；runner 参数同 v2（`--profile mixed --load 90 --scenario steady --api-key vx-dev --resume`）。
  - **OOD 子矩阵（v2 漏写）**：4 策略 FQCN（StockFCFSShim、PureLTRScheduler、TailSafeScheduler、GatedHybridScheduler）× `--profile ood --repeats 3` = 12 runs，workload 用 ood.v2.jsonl，输出 `runs/matrix-ood/`。
  - 网关开销：`scripts/run_gateway_overhead.py`（成对重放 harness，`gateway_overhead.py:54`）独立一步，不混进策略矩阵。
  - **parity gate（v3 只"都跑"不判定）**：mixed 矩阵跑完后强制 `python3 scripts/check_fcfs_parity.py --stock runs/matrix/stock_fcfs.json --shim runs/matrix/<StockFCFSShim>.json --output runs/matrix/parity.json`（该 CLI 超 tolerance 会 exit 1，`check_fcfs_parity.py:22`）；**parity 失败不得写 DONE**。
  - 每 run 前写 run-manifest：两仓 `git rev-parse HEAD`、gateway-pin、mapping_version、`approximation_notice`、quantile manifest sha256、decision 超时实测值。
  - **preflight 硬门槛（矩阵前必过，不可 skip）**：`python -c "import vllm"` 成功 + `pytest tests/test_vllm_protocol_seam.py -q` 全 PASS（importorskip 在服务器上不可能触发 skip，preflight 先验证 import 保证这一点）。
  - 末尾 tar → `oss cp` → `oss ls` 读回校验 → DONE。
- [ ] **Step 7b: `compute_rental_budget.sh`（8 小时预算 gate，租卡前用假设 capacity 区间跑一次、租卡日校准后用真值再跑一次）**：**逐阶段计时，缺一不可**——环境/模型下载、restore、补 replay 3 行、quantile 构建、decision latency 实测（200 次）、协议接缝 + 2 请求 E2E 门槛、saturation calibration（6 档网格 × 120 请求）、mixed 矩阵（7 档 × 3 repeats，mixed 行数）、OOD 子矩阵（4 档 × 3 repeats，ood 行数）、gateway overhead 成对重放、每档模型重启 ≈3min、上传 ≈10min。**steady@90 单 run 请求时间按 `N/(0.9×capacity_rps)` 估算**（`runner.py:245` 的到达率语义）；租卡前用有据容量下界 0.75 rps 当最坏情形（3090 实测 202 tok/s 对折；用户决策 2026-07-19 预算门槛 6h→8h 保统计规模）。输出 `rental-budget.json`（逐阶段 + 总计）；**总计 > 7.25h → exit 1**，45 分钟只作为重试余量不摊入任何阶段。裁剪顺序写死：先砍 OOD repeats 3→2，再砍 mixed 非核心档（保 fcfs/pure_ltr/tail_safe/gated_hybrid）。
- [ ] **Step 8: `shellcheck scripts/server/*.sh` 零 error**；commit。

---

### Task 7: 租卡前冒烟 — CPU 全链 + 真 vLLM 协议接缝

**Files:**
- Create: `scripts/fake_vllm_server.py`、`scripts/smoke_local_chain.sh`
- Create: `tests/test_vllm_protocol_seam.py`（`pytest.importorskip("vllm")`）

**冒烟分两层（fake 引擎测不出 bool coercion——v2 第 4 号阻断的教训）：**
1. **CPU 全链（Mac）**：真 gateway 分支 build + 真 decision service（bert，`LTR_DECISION_TIMEOUT_MS=2000`）+ **runner 内部函数直驱** + 假 vLLM。**注意：runner CLI 起手就校验本机装了 vLLM distribution（`runner.py:607`），Mac 两个环境都没有 vLLM——所以 Mac 冒烟不走 runner CLI**，改用新驱动脚本 `scripts/smoke_chain_driver.py`：直接 import `stream_completion`/`run_replay`/`make_chat_payload` 发 20 请求（绕过 distribution guard 的唯一诚实方式是不进 `run_benchmark`，不许伪造 distribution metadata）。断言同前（20/20、reliable 带 estimate、`enable_thinking=false` 透传、`ltr_tool_schema` 已剥、**`prediction_reliable` 是数字 1 不是 bool**）。runner CLI 全流程留给租卡日（服务器有 vLLM）。
2. **真协议接缝（201 或租卡日 step 0）**：`tests/test_vllm_protocol_seam.py` —— 用 pinned vLLM 构造完整 `ChatCompletionRequest(model=..., messages=[...], vllm_xargs={"prediction_reliable": 1, "workflow_estimated_tokens": 512, ...})`，调用**真实签名** `request.to_sampling_params(max_tokens=4096, default_sampling_params={})`（该方法不能空参调用，pinned protocol.py 为准；若签名仍不符，以 pinned 源码实际参数为准修测试而不是改产品码）→ 断言 `extra_args` 值类型（int 不是 bool）→ 喂 `GatewayMetadataPredictor` 断言 reliable 路径成立。优先在 201（`ssh -p 2222 alex@192.168.8.201` venv 装 `vllm==0.24.0` CPU）跑通；装不动则作为**租卡日矩阵前强制 step 0**。
3. **租卡日矩阵前门槛（≥2 条请求——`reorder_request_queue` 在 len<2 时 early-return，单请求进不了重排；Task 3 已让 early-return 前也写审计日志，但重排序证据仍需 ≥2）**：并发发 2 条真 Qwen 请求过全链 → `order.jsonl` 出现两个 request_id 且 `predictions` 里两条都 `ood == false`（**用 ood 判可靠，不用 `score != 1.0`——合法的 4096-token estimate 的 score 恰好是 1.0**），才准开矩阵。

- [ ] Step 1: `fake_vllm_server.py`（capture + chat SSE；对 `prediction_reliable` 类型断言 int）。
- [ ] Step 2: `scripts/smoke_chain_driver.py`（runner 内部函数直驱 20 请求）+ `smoke_local_chain.sh`（编排器持 EXIT trap）。**冒烟用的 quantile manifest = 明确标注的 fixture**（`model_version="smoke-fixture"`，合成 percentile 值）——生产 manifest 必须等租卡日补 replay 3 行 error 后由 merge + builder 生成（已实测：本地 merge 正确 NO-GO，报出 3 个缺长度 sample），不伪造 sample_count。冒烟只验链路集成，不验映射数值。
- [ ] Step 3: `test_vllm_protocol_seam.py` 写好并在 201 尝试执行；结果（跑通/装不动）如实记录进 README。
- [ ] Step 4: 证据归档 `runs/smoke-local-chain/`；commit。

---

### Task 8: 就绪重审（fresh Codex，GO/NO-GO）

- [ ] Step 1: 全量测试：`rtk proxy python3 -m pytest tests/ -q` + `GOTOOLCHAIN=auto go test ./internal/...`。
- [ ] Step 2: 新开 Codex 会话（不 resume），输入 = 本计划 + 两仓 `git log --oneline` + smoke 证据，逐项核对 11 项 NO-GO 缺口 + 本轮 15 项阻断，判决写 `docs/superpowers/plans/2026-07-19-readiness-verdict.md`。
- [ ] Step 3: NO-GO → 修 → 复审；GO → 租卡。

---

## 执行编排与工时（诚实版，v3 比 v2 增加 repack/协议接缝/parity）

| Task | 估时 | 会话 |
|---|---:|---|
| 1 Go adapter（含验证/白名单） | 5–7h | A |
| 2 decision service | 2–3h | B |
| 3 gateway predictor | 1.5–2h | B |
| 4 tier2 转正+依赖拆分 | 1.5–2h | B |
| 5 runner chat+history 迁移 | 3–4h | B（依赖 4） |
| 6 服务器脚本+repack | 4–6h | A（依赖 1,2；repack 可先行） |
| 7 双层冒烟 | 2.5–4h | A+B 汇合 |
| 8 重审 | 1–2h | fresh |
| **合计** | **21–30h** | 两会话并行墙钟 ≈ **13–18h** |

**时间表：** 周六全天 + 周日大半天 → **租卡现实目标 = 周一**（周日深夜是激进值）；GPU 矩阵 6h（mixed 7 档 21 runs + OOD 12 runs + overhead 对照）。

## v3 复核 8 项 → v4 落点

1 [P0] serve 参数缺打标环境 → Global Constraints 新 bullet + `launch_vllm.sh` 补全五参数。2 [P0] labels-merged 造不出来 → Step 1 定位/重建 `tier2-toolace-sample-6000.jsonl`（SHA `ee5a5889...` 校验）+ Step 1b 两文件 join + 租卡日补 replay 3 行 error + exact-count 规则修正（censored 有长度，缺的是 error）。3 [P0] bool 破坏 int 合同 → transport 输出 `RELIABLE/UNRELIABLE` int + predictor isinstance 排除 bool + `True` 回退反例测试。4 [P0] 协议接缝/门槛不可执行 → `to_sampling_params(max_tokens=4096, default_sampling_params={})` 真签名；Mac 冒烟不走 runner CLI（distribution guard，runner.py:607）改内部函数直驱；order log 在 len<2 也写 + `predictions` 审计字段；租卡门槛改 ≥2 并发请求。5 [P1] parity 无判定 → `check_fcfs_parity.py` 强制 gate，失败不 DONE。6 [P1] 超时实测法不安全 → 真 HTTP、并发 8、200 次、`max(2000, ceil(1.25*p99))`。7 [P1] mapper provenance 未定义 → fixture 代码显式给出、sha=manifest bytes SHA-256、构造参数、reliable/unreliable 都带三字段、transport validator 校验、`vllm_scheduler.py` 补 `import json`。8 [P1] 6h 预算无 gate → `compute_rental_budget.sh`（>5.25h exit 1，裁剪顺序写死）。

## v2 REJECT 15 项阻断 → v3 落点

1 模块名 `veloxmesh` → Task 1 全部 import 修正。2 客户端 xargs 无白名单 → 白名单 {ltr_kind, ltr_category} + 恶意字段测试。3 Go 响应验证不足 → contract switch 全套 + 违约 fail-open 测试。4 bool 不在 vllm_xargs 类型集 → 全局合同 int 0/1 + predictor `==1` + 协议接缝测试（Task 7）。5 300ms 超时未实测 → 默认 2000ms + 服务器实测写 manifest。6 FakeRequest 四参数/未定义变量 → 真实 dataclass 签名 + 完整 order-log 测试代码。7 torch pin 冲突 → requirements/train.txt 与 server.in 拆分。8 OOD wrapped 格式测错 → wrapped 测试 + `_toolace_tools` passthrough。9 history 丢失 → workload v2 含 history（必填）+ payload/合同测试。10 launcher EXIT trap 自杀 → 编排器持 EXIT，launcher 仅 INT/TERM。11 scheduler 参数 FQCN → 双侧同 FQCN、循环取自 `SCHEDULER_CLASS_TO_POLICY`。12 OSS 对象名/布局失实 → inventory_and_repack 以 T7 实测为准（已核：只有两个 tar）。13 quantile 输入未定义 → `labels-merged-6k.jsonl` 合并步骤 + 真实 builder CLI。14 fcfs-direct 语义错误 → 恢复 stock_fcfs parity 档 + `run_gateway_overhead.py`。15 fake 引擎测不出协议接缝 → `test_vllm_protocol_seam.py` + 租卡日 step 0 真请求门槛。

另：Task 2 使用真实 helper `valid_request`/`make_app`（:15/:48）并同步更新受 4096 影响的既有精确匹配测试；`MAX_ESTIMATED_TOKENS` 唯一化到 `contracts.py`；decision response 与 run manifest 携带 mapping_version/approximation_notice/manifest sha；Qwen `--local-dir` 与 `--dtype bfloat16` 显式；gateway fork 按 commit SHA 锁定（`manifest/gateway-pin.txt`）；OOD 子矩阵（4 策略×3 repeats）与饱和标定失败规则已写死。
