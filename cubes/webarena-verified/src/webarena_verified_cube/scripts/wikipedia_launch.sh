#!/usr/bin/env bash
# Launch script for webarena-verified Wikipedia (Kiwix) container.
# Volumes are pre-populated by VolumeSpec during provision().
set -euo pipefail

docker run -d \
    --name webarena_wikipedia \
    -p 8888:8080 \
    -p 8889:8874 \
    -v webarena_wikipedia_data:/data \
    ghcr.io/kiwix/kiwix-serve:3.8.0 \
    /data/wikipedia_en_all_maxi_2022-05.zim

healthy=0
for i in $(seq 1 60); do
    curl -sf http://localhost:8888/ > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: wikipedia did not become healthy after 120s" >&2
    exit 1
fi
