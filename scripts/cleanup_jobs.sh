#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_DIR="data/jobs"
MODE="${1:-all}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "No $TARGET_DIR directory; nothing to clean."
  exit 0
fi

if [[ "$MODE" == "all" ]]; then
  echo "Cleaning all job dirs under $TARGET_DIR ..."
  find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  echo "Done."
  exit 0
fi

if [[ "$MODE" =~ ^[0-9]+$ ]]; then
  echo "Cleaning job dirs older than $MODE day(s) under $TARGET_DIR ..."
  find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +"$MODE" -exec rm -rf {} +
  echo "Done."
  exit 0
fi

cat <<'EOF'
Usage:
  ./scripts/cleanup_jobs.sh all
  ./scripts/cleanup_jobs.sh <days>

Examples:
  ./scripts/cleanup_jobs.sh all
  ./scripts/cleanup_jobs.sh 7
EOF
