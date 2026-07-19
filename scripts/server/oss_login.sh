#!/usr/bin/env bash
# Non-interactive HyCloud OSS login.
# One-time setup (done by the human, never by an agent): create ~/.oss_env with
#   export OSS_USER='<exact username typed at the interactive prompt>'
#   export OSS_PW='<password>'
# chmod 600 ~/.oss_env. The CLI reads credentials from /dev/tty, so this
# script drives it through expect; values are shell-expanded, never printed.
set -euo pipefail

ENV_FILE="${OSS_ENV_FILE:-$HOME/.oss_env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
: "${OSS_USER:?OSS_USER is not set — create ~/.oss_env first}"
: "${OSS_PW:?OSS_PW is not set — create ~/.oss_env first}"
export OSS_USER OSS_PW
command -v expect >/dev/null || { echo "expect is required" >&2; exit 1; }

# shellcheck disable=SC2016  # $env(...) is Tcl, expanded by expect not bash
expect -c '
set timeout 30
log_user 0
spawn oss login
expect -re "(?i)username"
send "$env(OSS_USER)\r"
expect -re "(?i)password"
send "$env(OSS_PW)\r"
expect eof
' >/dev/null

oss ls >/dev/null
echo "oss login OK (listing verified)"
