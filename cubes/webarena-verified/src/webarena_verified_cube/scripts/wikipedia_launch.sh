#!/usr/bin/env bash
# Launch script for webarena-verified Wikipedia (Kiwix) container.
# Volumes are pre-populated by VolumeSpec during provision().
set -euo pipefail

dump_wikipedia_debug() {
    echo "[wikipedia][debug] container status:"
    docker ps -a --filter name=webarena_wikipedia --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    echo "[wikipedia][debug] recent container logs:"
    docker logs --tail 200 webarena_wikipedia 2>&1 || true
    echo "[wikipedia][debug] /data listing + file details inside container:"
    docker exec webarena_wikipedia sh -lc \
        'ls -lah /data && stat /data/wikipedia_en_all_maxi_2022-05.zim || true' 2>&1 || true
    echo "[wikipedia][debug] kiwix/environment-control versions:"
    docker exec webarena_wikipedia sh -lc \
        'kiwix-serve --version || true; python3 -m environment_control.cli --help >/dev/null && python3 -c "import environment_control; print(environment_control.__version__)" || true' 2>&1 || true
}

docker run -d \
    --name webarena_wikipedia \
    -p 8888:8080 \
    -p 8889:8874 \
    -v webarena_wikipedia_data:/data \
    am1n3e/webarena-verified-wikipedia

healthy=0
for i in $(seq 1 60); do
    curl -sf http://localhost:8888/ > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: wikipedia did not become healthy after 120s" >&2
    dump_wikipedia_debug
    exit 1
fi
