#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <tag>"
  exit 1
fi

TAG="$1"
BASE_IMAGE="isaac-lab-base"
FINAL_IMAGE="nvcr.io/nvidian/octi-isaac-lab:${TAG}"

echo "▶️  Starting container service"
ISAACLAB_NOCACHE=1  ./docker/container.py start

echo "🏷  Tagging image as ${FINAL_IMAGE}"
docker tag "${BASE_IMAGE}" "${FINAL_IMAGE}"

echo "📤 Pushing to NGC: ${FINAL_IMAGE}"
/home/zhengyuz/ngc-cli/ngc registry image push "${FINAL_IMAGE}"
echo "📤 Image: ${FINAL_IMAGE} pushed to NGC: "

echo "✅ Done!"
