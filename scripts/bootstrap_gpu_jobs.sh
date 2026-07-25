#!/usr/bin/env bash
set -euo pipefail

while pgrep -f '^bash /hy-tmp/staging/scripts/setup_gpu_env.sh$' >/dev/null; do
  sleep 10
done

bash /hy-tmp/staging/scripts/verify_gpu_env.sh

nohup bash /hy-tmp/staging/scripts/run_lmcache_tier1.sh \
  > /hy-tmp/logs/lmcache-tier1.log 2>&1 < /dev/null &
echo "$!" > /hy-tmp/logs/lmcache-tier1.pid

nohup bash /hy-tmp/staging/scripts/run_gpu_overnight.sh \
  > /hy-tmp/logs/gpu-overnight.log 2>&1 < /dev/null &
echo "$!" > /hy-tmp/logs/gpu-overnight.pid
