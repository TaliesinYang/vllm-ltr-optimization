#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
GATEWAY_REPO="${GATEWAY_REPO:-$LTR_ROOT/VeloxMesh}"
GATEWAY_FORK_URL="${GATEWAY_FORK_URL:-https://github.com/TaliesinYang/VeloxMesh.git}"
PIN_FILE="$REPO_ROOT/scripts/server/manifest/gateway-pin.txt"
GO_VERSION="${GO_VERSION:-1.26.1}"

mkdir -p "$LTR_ROOT/bin"
if ! command -v go >/dev/null; then
  archive="/tmp/go${GO_VERSION}.linux-amd64.tar.gz"
  curl -fL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o "$archive"
  rm -rf "$LTR_ROOT/go"
  tar -xzf "$archive" -C "$LTR_ROOT"
  export PATH="$LTR_ROOT/go/bin:$PATH"
fi

if [[ ! -d "$GATEWAY_REPO/.git" ]]; then
  git clone "$GATEWAY_FORK_URL" "$GATEWAY_REPO"
fi
pin="$(tr -d '[:space:]' <"$PIN_FILE")"
git -C "$GATEWAY_REPO" fetch --all --tags
git -C "$GATEWAY_REPO" checkout --detach "$pin"
[[ "$(git -C "$GATEWAY_REPO" rev-parse HEAD)" == "$pin" ]] || { echo "gateway pin mismatch" >&2; exit 1; }
(
  cd "$GATEWAY_REPO"
  GOTOOLCHAIN=auto go test ./internal/ltr/
  GOTOOLCHAIN=auto go build -o "$LTR_ROOT/bin/gateway" ./cmd/gateway
)
echo "gateway built at pinned commit $pin"
