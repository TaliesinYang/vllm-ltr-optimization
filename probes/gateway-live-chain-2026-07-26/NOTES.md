# Live chain smoke — OpenCode → VeloxMesh gateway → Ollama (ticket #6)

**Date:** 2026-07-26 (evening 07-25 PT) · **Result: PASS** — a real OpenCode task completed
end-to-end through the gateway with tool calls executing (multi-turn Read/Write loop),
against a real model producing real completion tokens.

## Topology

```
OpenCode (Mac, vanilla config, 10 native tools)
  → VeloxMesh gateway  Mac 127.0.0.1:9100  (branch feat/ltr-decision-adapter, 888fba9)
    → SSH tunnel       Mac 127.0.0.1:11435 → 201 localhost:11434
      → Ollama 0.32.4  201 box (WSL2), qwen2.5:7b-instruct (4.7 GB), RTX 4090 Laptop 16 GB
```

## Evidence

- curl smoke through gateway (warm): **HTTP 200, 0.4 s end-to-end, `X-Queue-Wait-Ms: 0`**,
  request id `118f93cf…`, completion "OK" with usage reported.
- OpenCode task ("list markdown files, read README") — gateway log shows the agent loop:
  4 requests, all 200, latency 673 / 1320 / 3667 / 1641 ms (title-gen + 3 agent turns).
  Tool calls (`glob`→`read`→`write`) actually executed against the project dir.
- Model true completion tokens recorded in each response's `usage`.

## Launch runbook (reproduce)

```bash
# 201: rootless Ollama (no sudo) — binary from the tar.zst release asset
curl -sL -o ollama.tar.zst https://github.com/ollama/ollama/releases/download/v0.32.4/ollama-linux-amd64.tar.zst
tar --zstd -xf ollama.tar.zst && ln -sf $PWD/bin/ollama ~/bin/ollama
OLLAMA_HOST=0.0.0.0:11434 setsid nohup ~/bin/ollama serve &
~/bin/ollama pull qwen2.5:7b-instruct

# Mac: tunnel (WSL2 has no LAN portproxy for 11434 — 2222 is the only proxied port)
ssh -p 2222 -N -L 11435:localhost:11434 alex@192.168.8.201 &

# Mac: gateway (env-var provider config; binary: go build ./cmd/gateway)
GATEWAY_DATA_ADDR=127.0.0.1:9100 DEV_API_KEY=vx-dev DEFAULT_PROVIDER=openai-primary \
OPENAI_PRIMARY_BASE_URL=http://127.0.0.1:11435/v1 OPENAI_PRIMARY_API_KEY=unused \
OPENAI_PRIMARY_MODELS="qwen2.5:7b-instruct" OPENAI_PRIMARY_DEFAULT_MODEL="qwen2.5:7b-instruct" \
./bin/gateway

# Mac: OpenCode — project-local provider (never touch global config); -m flag REQUIRED
# opencode.json: provider "vx", baseURL http://127.0.0.1:9100/v1, apiKey vx-dev
XDG_CONFIG_HOME=<empty-dir> opencode run -m vx/qwen2.5:7b-instruct "<task>"
```

## Pitfalls burned (do not repeat)

1. **WSL2 networking**: 201's Ollama is unreachable from LAN (no Windows portproxy for
   11434); direct Mac→201:11434 hangs ~20 s then dies. Tunnel through the existing 2222.
2. **Thinking models blow the provider timeout**: qwen3:8b spends ~20 s+ thinking per
   reply; the env-var provider path hardcodes `Timeout: "30s"`
   (internal/config/config.go:127) → intermittent 504. Use a non-thinking model
   (qwen2.5:7b-instruct) or a config-file provider with a longer timeout.
3. **Cold load 504s**: first request after idle loads 4.7 GB into VRAM (~20–60 s) → 504
   through the gateway. Warm the model with a direct request before measuring anything.
4. **Vanilla tool set for small models**: the user's full config declares 170 tools
   (~147 KB ≈ 40k+ tokens of schema) — over qwen2.5:7b's practical context. The vanilla
   config (10 native tools, ~21 KB) is the right smoke configuration.

## What this unblocks

- Ticket #7 (T3): trace collection — put a logging reverse proxy in front of :9100 (echo
  server pattern) and run ~20–30 tasks × 2 configs to build the E5 replay workload.
- The gateway is no longer an unexercised liability: it has carried real agent traffic.
