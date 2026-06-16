#!/usr/bin/env bash
# Launch script for webarena-verified reddit container.
set -euo pipefail

docker run -d \
    --name webarena_reddit \
    -p 9999:80 \
    -p 9998:8877 \
    am1n3e/webarena-verified-reddit

healthy=0
for i in $(seq 1 60); do
    curl -sf http://localhost:9999/login > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: reddit did not become healthy after 120s" >&2
    exit 1
fi

# Reddit rate-limit relaxation is appended here from resources.py (_REDDIT_RELAX),
# shared with all_sites_launch.sh — see scripts/reddit_relax_limits.sh.
