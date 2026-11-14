#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

preview_ip=${1:-$(ipconfig getifaddr en0 || true)}
if [ -z "$preview_ip" ]; then
    printf '%s\n' "Mac의 Wi-Fi 주소를 확인하고 sh scripts/run_mobile.sh <내부 IP>를 실행해 주세요." >&2
    exit 1
fi
PYTHONPATH=backend/src exec .venv/training/bin/python -m acne_care.api \
    --mobile-preview "$preview_ip" --port "${2:-8443}"
