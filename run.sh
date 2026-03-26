#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

MODE="${1:-help}"

case "$MODE" in
  web)
    shift || true
    exec uvicorn webapp.main:app --host 0.0.0.0 --port "${1:-8788}"
    ;;
  cli)
    shift || true
    exec python watermark_remover.py "$@"
    ;;
  help|*)
    cat <<'EOF'
Usage:
  ./run.sh web [port]
  ./run.sh cli --input 2.png --output 2_out.png

Examples:
  ./run.sh web
  ./run.sh cli --input 2.png --output 2_out.png --mask-output 2_mask.png
EOF
    ;;
esac
