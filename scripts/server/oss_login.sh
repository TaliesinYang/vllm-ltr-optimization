#!/usr/bin/env bash
# Non-interactive HyCloud OSS login from environment credentials.
# One-time setup (done by the human, never by an agent):
#   export HY_OSS_AK_ID=...      # AccessKey ID from the HyCloud console
#   export HY_OSS_AK_SECRET=...  # AccessKey secret
# Then any session (Mac or rental box with the vars exported) can run this.
set -euo pipefail

: "${HY_OSS_AK_ID:?HY_OSS_AK_ID is not set — export it in your shell profile first}"
: "${HY_OSS_AK_SECRET:?HY_OSS_AK_SECRET is not set — export it in your shell profile first}"

oss login -i="$HY_OSS_AK_ID" -k="$HY_OSS_AK_SECRET"
oss ls >/dev/null
echo "oss login OK (listing verified)"
