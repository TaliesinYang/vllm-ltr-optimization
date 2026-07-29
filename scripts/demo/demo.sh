#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/alex/develop/vllm-ltr-optimization"
GATEWAY_ROOT="/Users/alex/develop/VeloxMesh"
DECISION_PYTHON="$REPO_ROOT/.worktrees/final-training-artifacts/.venv/bin/python"
DECISION_LOG="/tmp/decision-service.log"
DECISION_TRAIL="/tmp/decisions.jsonl"
GATEWAY_LOG="/tmp/gateway.log"
DASHBOARD_LOG="/tmp/dashboard.log"
TUNNEL_LOG="/tmp/ollama-tunnel.log"
MODEL="qwen2.5:7b-instruct"
S4_PAYLOAD='{
  "schema_version": "1.0",
  "model_id": "qwen2.5:7b-instruct",
  "request_age_ms": 0,
  "generation_controls": {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "max_tokens": 512
  },
  "request_id": "demo-s4",
  "decision_id": "demo-s4-d",
  "messages": [
    {
      "role": "user",
      "content": "Find every markdown file in this project and read the README."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {
          "type": "object",
          "properties": {
            "pattern": {
              "type": "string"
            }
          },
          "required": ["pattern"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read a file from disk.",
        "parameters": {
          "type": "object",
          "properties": {
            "filePath": {
              "type": "string"
            }
          },
          "required": ["filePath"]
        }
      }
    }
  ]
}'

fail() {
  local step="$1"
  local hint="$2"
  printf 'ERROR [%s]: %s\n' "$step" "$hint" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    fail "prerequisite" "Missing command '$1'. Install it, then rerun this script."
}

require_file() {
  [[ -f "$1" ]] ||
    fail "prerequisite" "Missing required file: $1"
}

port_is_listening() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

http_is_up() {
  curl --fail --silent --show-error --max-time 3 "$1" >/dev/null 2>&1
}

wait_for_http() {
  local step="$1"
  local url="$2"
  local attempts="$3"
  local hint="$4"
  local attempt

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if http_is_up "$url"; then
      return 0
    fi
    sleep 1
  done
  fail "$step" "$hint"
}

start_tunnel_if_needed() {
  if port_is_listening 11435; then
    printf 'SKIP tunnel: port 11435 already listening\n'
    return
  fi

  printf 'START tunnel: localhost:11435 -> 192.168.8.201:11434\n'
  nohup ssh -p 2222 \
    -o ConnectTimeout=8 \
    -o BatchMode=yes \
    -o ServerAliveInterval=30 \
    -N -L 11435:localhost:11434 alex@192.168.8.201 \
    >>"$TUNNEL_LOG" 2>&1 &

  wait_for_http \
    "SSH tunnel" \
    "http://127.0.0.1:11435/api/tags" \
    20 \
    "Tunnel did not expose Ollama. Check $TUNNEL_LOG and SSH access to alex@192.168.8.201:2222."
}

warm_ollama() {
  local payload
  payload="{\"model\":\"$MODEL\",\"prompt\":\"Reply OK\",\"stream\":false,\"options\":{\"num_predict\":1}}"
  printf 'WARM Ollama: loading %s (cold load may take 20-60s)\n' "$MODEL"
  curl --fail --silent --show-error --max-time 90 \
    -H "Content-Type: application/json" \
    --data-binary "$payload" \
    "http://127.0.0.1:11435/api/generate" >/dev/null ||
    fail "Ollama warmup" "Model $MODEL did not answer within 90s. Check the tunnel and run 'ollama pull $MODEL' on 192.168.8.201."
}

start_decision_if_needed() {
  if port_is_listening 9200; then
    printf 'SKIP decision service: port 9200 already listening\n'
  else
    printf 'START decision service: 127.0.0.1:9200\n'
    (
      cd "$REPO_ROOT"
      DECISION_LOG_PATH="$DECISION_TRAIL" PYTHONPATH="$PWD" \
        nohup "$DECISION_PYTHON" scripts/run_decision_service.py \
        --host 127.0.0.1 --port 9200 --predictor bert \
        --checkpoint checkpoints_best_predictor \
        --quantile-manifest runs/full-stack-demo-2026-07-26/rank_quantiles.json \
        --max-concurrency 8 --reliability-threshold 0.5 \
        >>"$DECISION_LOG" 2>&1 &
    )
  fi

  wait_for_http \
    "decision service" \
    "http://127.0.0.1:9200/healthz" \
    180 \
    "Decision service is not ready. Check $DECISION_LOG; BERT loading can take up to 3 minutes."
}

start_gateway_if_needed() {
  if port_is_listening 9100; then
    printf 'SKIP gateway: port 9100 already listening\n'
  else
    printf 'START gateway: 127.0.0.1:9100\n'
    (
      cd "$GATEWAY_ROOT"
      GATEWAY_DATA_ADDR=127.0.0.1:9100 DEV_API_KEY=vx-dev DEFAULT_PROVIDER=openai-primary \
      OPENAI_PRIMARY_BASE_URL=http://127.0.0.1:11435/v1 OPENAI_PRIMARY_API_KEY=unused \
      OPENAI_PRIMARY_MODELS="$MODEL" OPENAI_PRIMARY_DEFAULT_MODEL="$MODEL" \
      LTR_DECISION_ENDPOINT=http://127.0.0.1:9200 LTR_DECISION_TIMEOUT_MS=2000 \
        nohup ./bin/gateway >>"$GATEWAY_LOG" 2>&1 &
    )
  fi

  wait_for_http \
    "gateway" \
    "http://127.0.0.1:9100/healthz" \
    60 \
    "Gateway is not healthy. Check $GATEWAY_LOG and verify /Users/alex/develop/VeloxMesh/bin/gateway exists."
}

start_dashboard_if_needed() {
  if port_is_listening 9310; then
    printf 'SKIP dashboard: port 9310 already listening\n'
  else
    printf 'START dashboard: 0.0.0.0:9310\n'
    (
      cd "$REPO_ROOT"
      DASHBOARD_HOST=0.0.0.0 \
        nohup "$DECISION_PYTHON" scripts/demo/dashboard.py \
        >>"$DASHBOARD_LOG" 2>&1 &
    )
  fi

  wait_for_http \
    "dashboard" \
    "http://127.0.0.1:9310/data.json" \
    30 \
    "Dashboard is not serving data. Check $DASHBOARD_LOG and port 9310."
}

warm_and_measure_decision_probe() {
  local elapsed_seconds
  curl --fail --silent --show-error --max-time 10 \
    -H "Content-Type: application/json" \
    --data-binary "$S4_PAYLOAD" \
    --output /tmp/demo-s4-warmup-response.json \
    "http://127.0.0.1:9200/v1/decision" ||
    fail "S4 probe warmup" "S4 decision failed. Check $DECISION_LOG."

  elapsed_seconds="$(
    curl --fail --silent --show-error --max-time 10 \
      -H "Content-Type: application/json" \
      --data-binary "$S4_PAYLOAD" \
      --output /tmp/demo-s4-hot-response.json \
      --write-out "%{time_total}" \
      "http://127.0.0.1:9200/v1/decision"
  )" || fail "S4 probe measurement" "Hot S4 decision failed after warmup. Check $DECISION_LOG."
  awk -v seconds="$elapsed_seconds" 'BEGIN { printf "%.2f", seconds * 1000 }'
}

service_status() {
  local url="$1"
  if http_is_up "$url"; then
    printf 'UP'
  else
    printf 'DOWN'
  fi
}

print_health_table() {
  local decision_ms="$1"
  local default_interface
  local lan_ip
  lan_ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
  if [[ -z "$lan_ip" ]]; then
    default_interface="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
    if [[ -n "$default_interface" ]]; then
      lan_ip="$(ipconfig getifaddr "$default_interface" 2>/dev/null || true)"
    fi
  fi

  printf '\n%-18s %-6s %s\n' "SERVICE" "STATE" "ENDPOINT"
  printf '%-18s %-6s %s\n' "SSH tunnel/Ollama" "$(service_status http://127.0.0.1:11435/api/tags)" "127.0.0.1:11435"
  printf '%-18s %-6s %s\n' "Decision service" "$(service_status http://127.0.0.1:9200/healthz)" "http://127.0.0.1:9200"
  printf '%-18s %-6s %s\n' "Gateway" "$(service_status http://127.0.0.1:9100/healthz)" "http://127.0.0.1:9100"
  printf '%-18s %-6s %s\n' "Dashboard" "$(service_status http://127.0.0.1:9310/data.json)" "http://127.0.0.1:9310"
  printf '\nDashboard local: http://127.0.0.1:9310\n'
  if [[ -n "$lan_ip" ]]; then
    printf 'Dashboard LAN:   http://%s:9310\n' "$lan_ip"
  else
    printf 'Dashboard LAN:   unavailable (en0 has no IPv4 address)\n'
  fi
  printf 'S4 decision hot: %s ms\n' "$decision_ms"
}

main() {
  require_command curl
  require_command lsof
  require_command ssh
  require_command awk
  require_command route
  require_file "$DECISION_PYTHON"
  require_file "$REPO_ROOT/scripts/demo/dashboard.py"
  require_file "$GATEWAY_ROOT/bin/gateway"

  start_tunnel_if_needed
  warm_ollama
  start_decision_if_needed
  start_gateway_if_needed
  start_dashboard_if_needed

  local decision_ms
  decision_ms="$(warm_and_measure_decision_probe)"
  print_health_table "$decision_ms"
}

main "$@"
