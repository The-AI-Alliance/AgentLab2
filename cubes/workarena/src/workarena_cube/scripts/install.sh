#!/usr/bin/env bash
# Install Playwright browser binaries (idempotent — skips already-installed browsers).
# Called automatically by BenchmarkConfig.install().
set -euo pipefail

playwright install chromium
