#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="asr-legacy"
CONTAINER_NAME="asr-legacy"

docker build -t "$IMAGE_NAME" .

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --init \
  --network host \
  --name "$CONTAINER_NAME" \
  -v "$(pwd)":/app \
  "$IMAGE_NAME" \
  sleep infinity
