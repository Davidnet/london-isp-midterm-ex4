#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker build -q -t asr-report . >/dev/null
# the pandoc image's ENTRYPOINT is pandoc itself
docker run --rm -v "$PWD:/work" -w /work asr-report \
  report.md -o report.pdf --pdf-engine=xelatex
words=$(docker run --rm -v "$PWD:/work" -w /work asr-report report.md -t plain | wc -w | tr -d ' ')
echo "report.pdf built (≈ ${words} words incl. captions)"
