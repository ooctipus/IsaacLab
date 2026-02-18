#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build and Push Script for Isaac Lab Docker Images (Newton)
# =============================================================================
#
# Uses a two-stage caching strategy:
#   1. isaac-lab-deps:<hash>  - Base + pip dependencies (cached by dep hash)
#   2. isaac-lab-base         - Deps + source code (final image)
#
# If dependency files haven't changed, skips pip install entirely.
#
# Usage:
#   ./buildnpush.sh <tag> [options]
#   ./buildnpush.sh enter <tag>       # Enter a container for the given tag
#   ./buildnpush.sh --status          # Show images and disk usage
#   ./buildnpush.sh --clean           # Remove old deps images (keep current)
#   ./buildnpush.sh --clean-all       # Remove ALL Isaac Lab images
#
# Build Options:
#   --force           Force full rebuild (ignore all caches)
#   --force-deps      Force rebuild dependencies (but use Docker cache)
#   --skip-push       Build only, don't push to NGC
#   -h, --help        Show this help message
#
# Examples:
#   ./buildnpush.sh v1.0              # Smart build (skip deps if unchanged)
#   ./buildnpush.sh v1.0 --force      # Full rebuild
#   ./buildnpush.sh enter v1.0        # Enter container for v1.0
#   ./buildnpush.sh --status          # Check current images
#   ./buildnpush.sh --clean           # Cleanup old deps caches
# =============================================================================

show_help() {
  head -33 "$0" | tail -31
  exit 0
}

image_exists() {
  docker image inspect "$1" &>/dev/null
}

# =============================================================================
# Enter Container Command
# =============================================================================
enter_container() {
  local tag="$1"
  local mount_local="$2"
  
  if [ -z "$tag" ]; then
    echo "Error: Tag required for 'enter' command"
    echo "Usage: ./buildnpush.sh enter <tag> [--mount]"
    echo "  --mount  Mount local directory to /local (container's isaaclab untouched)"
    exit 1
  fi
  
  # Try NGC image first, fall back to local
  local image="nvcr.io/nvidian/octi-isaac-lab:${tag}"
  if ! image_exists "$image"; then
    image="isaac-lab-base:${tag}"
    if ! image_exists "$image"; then
      image="isaac-lab-base"
      if ! image_exists "$image"; then
        echo "Error: No image found for tag '${tag}'"
        echo "Available images:"
        docker images --format "  {{.Repository}}:{{.Tag}}" | grep -E "(isaac-lab|octi-isaac-lab)" || echo "  (none)"
        exit 1
      fi
    fi
  fi
  
  local mount_args=""
  if [ "$mount_local" = "--mount" ]; then
    mount_args="-v $(pwd):/local:rw"
    echo "🚀 Entering container: ${image} (local mounted at /local)"
  else
    echo "🚀 Entering container: ${image}"
  fi
  
  docker run -it --rm \
    --gpus all \
    --network host \
    -e "ACCEPT_EULA=Y" \
    -e "OMNI_KIT_ALLOW_ROOT=1" \
    $mount_args \
    --entrypoint /bin/bash \
    "${image}"
  exit 0
}

# =============================================================================
# Cleanup and Status Commands
# =============================================================================
show_status() {
  echo "📊 Isaac Lab Docker Images Status (Newton)"
  echo "==========================================="
  echo ""
  
  # Current deps hash
  local current_hash=$(compute_deps_hash)
  echo "Current deps hash: ${current_hash}"
  echo ""
  
  # List all isaac-lab images
  echo "Images:"
  docker images --format "  {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | grep -E "^  (isaac-lab|nvcr.io/nvidian/octi-isaac-lab)" || echo "  (none)"
  echo ""
  
  # Total disk usage
  echo "Disk Usage:"
  docker system df --format "  Images: {{.Size}}" 2>/dev/null | head -1 || true
  echo ""
  
  # Highlight current deps image
  if image_exists "isaac-lab-deps:${current_hash}"; then
    echo "✓ Current deps image exists: isaac-lab-deps:${current_hash}"
  else
    echo "✗ No deps image for current hash: ${current_hash}"
  fi
  
  exit 0
}

clean_old_deps() {
  echo "🧹 Cleaning old deps images..."
  echo ""
  
  local current_hash=$(compute_deps_hash)
  local current_image="isaac-lab-deps:${current_hash}"
  local removed=0
  
  # List all isaac-lab-deps images
  local deps_images=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^isaac-lab-deps:" || true)
  
  if [ -z "$deps_images" ]; then
    echo "No deps images found."
    exit 0
  fi
  
  echo "Current hash: ${current_hash}"
  echo ""
  
  for img in $deps_images; do
    if [ "$img" = "$current_image" ]; then
      echo "  ✓ Keeping: $img (current)"
    else
      echo "  🗑 Removing: $img"
      docker rmi "$img" 2>/dev/null || echo "    (in use, skipped)"
      ((removed++)) || true
    fi
  done
  
  echo ""
  echo "Removed $removed old deps image(s)."
  
  # Also prune dangling images
  echo ""
  echo "Pruning dangling images..."
  docker image prune -f
  
  exit 0
}

clean_all() {
  echo "🧹 Removing ALL Isaac Lab Docker images..."
  echo ""
  echo "⚠️  This will remove:"
  echo "    - All isaac-lab-base images"
  echo "    - All isaac-lab-deps:* images"
  echo "    - All nvcr.io/nvidian/octi-isaac-lab:* images"
  echo ""
  read -p "Are you sure? [y/N] " -n 1 -r
  echo ""
  
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
  
  echo ""
  
  # Remove all matching images
  docker images --format '{{.Repository}}:{{.Tag}}' | grep -E "^(isaac-lab-base|isaac-lab-deps:|nvcr.io/nvidian/octi-isaac-lab:)" | while read img; do
    echo "  🗑 Removing: $img"
    docker rmi "$img" 2>/dev/null || echo "    (failed, may be in use)"
  done
  
  echo ""
  echo "Pruning dangling images..."
  docker image prune -f
  
  echo ""
  echo "Done!"
  exit 0
}

# =============================================================================
# Compute hash of dependency files
# =============================================================================
compute_deps_hash() {
  # Hash all files that affect pip dependencies
  find source -name "setup.py" -o -name "extension.toml" -o -name "requirements.txt" -o -name "pyproject.toml" 2>/dev/null | \
    sort | \
    xargs cat 2>/dev/null | \
    md5sum | \
    cut -d' ' -f1 | \
    head -c 12
}

# =============================================================================
# Compute hash of source code files
# =============================================================================
compute_source_hash() {
  # Hash all Python source files that would trigger reinstall
  find source -type f \( -name "*.py" -o -name "*.pyx" -o -name "*.pxd" \) ! -path "*/__pycache__/*" ! -path "*/.git/*" 2>/dev/null | \
    sort | \
    xargs md5sum 2>/dev/null | \
    md5sum | \
    cut -d' ' -f1 | \
    head -c 12
}

# =============================================================================
# Parse Arguments
# =============================================================================
FORCE_REBUILD=0
FORCE_DEPS=0
SKIP_PUSH=0
TAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --status)
      show_status
      ;;
    --clean)
      clean_old_deps
      ;;
    --clean-all)
      clean_all
      ;;
    enter)
      shift
      enter_container "${1:-}" "${2:-}"
      ;;
    --force)
      FORCE_REBUILD=1
      shift
      ;;
    --force-deps)
      FORCE_DEPS=1
      shift
      ;;
    --skip-push)
      SKIP_PUSH=1
      shift
      ;;
    -h|--help)
      show_help
      ;;
    -*)
      echo "Unknown option: $1"
      show_help
      ;;
    *)
      if [ -z "$TAG" ]; then
        TAG="$1"
      else
        echo "Unexpected argument: $1"
        show_help
      fi
      shift
      ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "Error: Tag is required"
  show_help
fi

# =============================================================================
# Configuration
# =============================================================================
DEPS_HASH=$(compute_deps_hash)
SOURCE_HASH=$(compute_source_hash)
DEPS_IMAGE="isaac-lab-deps:${DEPS_HASH}"
SOURCE_IMAGE="isaac-lab-source:${SOURCE_HASH}"
BASE_IMAGE="isaac-lab-base"
FINAL_IMAGE="nvcr.io/nvidian/octi-isaac-lab:${TAG}"

echo "🔍 Dependency Analysis..."
echo "   Deps Hash:  ${DEPS_HASH}"
echo "   Source Hash: ${SOURCE_HASH}"

# =============================================================================
# Determine Build Strategy
# =============================================================================
SKIP_DEPS=0
SKIP_SOURCE=0
USE_CACHE=1
REASON=""

# Check if ANY isaac-lab-deps:* images exist (indicates we're already using new system)
any_deps_image_exists() {
  docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^isaac-lab-deps:" 2>/dev/null
}

if [ "$FORCE_REBUILD" -eq 1 ]; then
  SKIP_DEPS=0
  SKIP_SOURCE=0
  USE_CACHE=0
  REASON="forced full rebuild (--force)"
elif [ "$FORCE_DEPS" -eq 1 ]; then
  SKIP_DEPS=0
  SKIP_SOURCE=0
  USE_CACHE=1
  REASON="forced deps rebuild (--force-deps)"
elif image_exists "$SOURCE_IMAGE"; then
  # Both deps and source are cached - can use source image directly
  SKIP_DEPS=1
  SKIP_SOURCE=1
  USE_CACHE=1
  REASON="fully cached (deps:${DEPS_HASH}, source:${SOURCE_HASH})"
  echo "   ✓ Cached source image found: ${SOURCE_IMAGE}"
elif image_exists "$DEPS_IMAGE"; then
  # Exact deps hash match - use cached deps, but need to rebuild source
  SKIP_DEPS=1
  SKIP_SOURCE=0
  USE_CACHE=1
  REASON="deps cached, source changed (${DEPS_IMAGE})"
  echo "   ✓ Cached deps image found: ${DEPS_IMAGE}"
  echo "   ⚠ Source changed - will rebuild source layer"
elif ! any_deps_image_exists && image_exists "$BASE_IMAGE"; then
  # ONE-TIME MIGRATION: No deps images exist yet, but isaac-lab-base exists
  # This means we're migrating from old system to new system
  echo "   ⚡ One-time migration: tagging existing ${BASE_IMAGE} as deps cache..."
  docker tag "${BASE_IMAGE}" "${DEPS_IMAGE}"
  SKIP_DEPS=1
  SKIP_SOURCE=0
  USE_CACHE=1
  REASON="migrated existing image (one-time)"
  echo "   ✓ Tagged as ${DEPS_IMAGE}"
else
  # Deps changed (hash doesn't match any existing deps image)
  SKIP_DEPS=0
  SKIP_SOURCE=0
  USE_CACHE=1
  if any_deps_image_exists; then
    REASON="deps changed, full rebuild required"
    echo "   ⚠ Deps hash changed - full rebuild required"
  else
    REASON="first build (no existing images)"
  fi
  echo "   ✗ No cached deps for hash ${DEPS_HASH}"
fi

echo ""
echo "📋 Build Configuration:"
echo "   Tag:           ${TAG}"
echo "   Deps Hash:     ${DEPS_HASH}"
echo "   Source Hash:   ${SOURCE_HASH}"
echo "   Strategy:      ${REASON}"
echo "   Build Deps:    $([ "$SKIP_DEPS" -eq 0 ] && echo "YES" || echo "SKIP (cached)")"
echo "   Build Source:  $([ "$SKIP_SOURCE" -eq 0 ] && echo "YES" || echo "SKIP (cached)")"
echo "   Docker Cache:  $([ "$USE_CACHE" -eq 1 ] && echo "YES" || echo "NO")"
echo "   Push to NGC:   $([ "$SKIP_PUSH" -eq 0 ] && echo "YES" || echo "SKIP")"
echo ""

# =============================================================================
# Symlink Resolution (Docker COPY doesn't follow external symlinks)
# =============================================================================
ASSETS_DATA_DIR="source/isaaclab_assets/data"
SYMLINKS_FILE=$(mktemp)

resolve_symlinks() {
  echo "🔗 Resolving symlinks in ${ASSETS_DATA_DIR}..."
  if [ ! -d "${ASSETS_DATA_DIR}" ]; then
    echo "   Directory does not exist, skipping"
    return
  fi
  for item in "${ASSETS_DATA_DIR}"/*; do
    if [ -L "$item" ]; then
      target=$(readlink -f "$item")
      if [ -d "$target" ]; then
        echo "   📁 Resolving: $item -> $target"
        rm "$item"
        cp -rL "$target" "$item"
        echo "$item:$target" >> "$SYMLINKS_FILE"
      fi
    fi
  done
}

restore_symlinks() {
  if [ ! -s "$SYMLINKS_FILE" ]; then
    rm -f "$SYMLINKS_FILE"
    return
  fi
  echo "🔗 Restoring symlinks..."
  while IFS=: read -r item target; do
    echo "   🔗 Restoring: $item -> $target"
    rm -rf "$item"
    ln -s "$target" "$item"
  done < "$SYMLINKS_FILE"
  rm -f "$SYMLINKS_FILE"
}

trap restore_symlinks EXIT
resolve_symlinks

# =============================================================================
# Build Stage 1: Dependencies Image (if needed)
# =============================================================================
if [ "$SKIP_DEPS" -eq 0 ]; then
  echo "▶️  Building full image with dependencies..."
  
  # Build with pip install enabled
  if [ "$USE_CACHE" -eq 0 ]; then
    echo "🔨 Building Docker image (no cache)..."
    docker build --no-cache -f docker/Dockerfile.newton -t "${BASE_IMAGE}" .
  else
    echo "🔨 Building Docker image (with cache)..."
    docker build -f docker/Dockerfile.newton -t "${BASE_IMAGE}" .
  fi
  
  # Tag as deps image for future cache hits
  echo "🏷  Caching deps image as ${DEPS_IMAGE}"
  docker tag "${BASE_IMAGE}" "${DEPS_IMAGE}"
  
  # If source is also cached, tag as source image
  if [ "$SKIP_SOURCE" -eq 1 ]; then
    echo "🏷  Caching source image as ${SOURCE_IMAGE}"
    docker tag "${BASE_IMAGE}" "${SOURCE_IMAGE}"
  fi
elif [ "$SKIP_SOURCE" -eq 1 ]; then
  echo "▶️  Using fully cached image (deps + source)..."
  # Both deps and source are cached - just tag the source image as base
  docker tag "${SOURCE_IMAGE}" "${BASE_IMAGE}"
else
  echo "▶️  Using cached dependencies, copying source only..."
  
  # Build using lightweight source-only Dockerfile
  if [ ! -f "docker/Dockerfile.source-only.newton" ]; then
    echo "   Creating docker/Dockerfile.source-only.newton..."
    cat > docker/Dockerfile.source-only.newton << 'EOF'
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

# Source-only Dockerfile for Isaac Lab with Newton
# Builds FROM a cached deps image and only copies/installs source code

ARG DEPS_BASE_IMAGE
FROM ${DEPS_BASE_IMAGE}

ARG ISAACLAB_PATH_ARG=/workspace/isaaclab
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}

# Copy the Isaac Lab directory
COPY ../ ${ISAACLAB_PATH}

# Ensure isaaclab.sh has execute permissions
RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh

# Install Isaac Lab packages (no Isaac Sim)
# This layer will only rebuild when source code changes
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab && \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab_assets && \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab_tasks && \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab_newton && \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab_rl && \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab_experimental && \
    pip install -e ${ISAACLAB_PATH}/source/isaaclab_tasks_experimental

# Set working directory
WORKDIR ${ISAACLAB_PATH}
EOF
  fi
  
  # Load env vars from .env.base if it exists
  if [ -f "docker/.env.base" ]; then
    source docker/.env.base
  fi
  
  # Build using lightweight source-only Dockerfile
  docker build \
    -f docker/Dockerfile.source-only.newton \
    --build-arg DEPS_BASE_IMAGE="${DEPS_IMAGE}" \
    --build-arg ISAACLAB_PATH_ARG="${ISAACLAB_PATH:-/workspace/isaaclab}" \
    -t "${BASE_IMAGE}" \
    .
  
  # Cache the source image for future builds
  echo "🏷  Caching source image as ${SOURCE_IMAGE}"
  docker tag "${BASE_IMAGE}" "${SOURCE_IMAGE}"
fi

# =============================================================================
# Tag and Push
# =============================================================================
echo "🏷  Tagging image as ${FINAL_IMAGE}"
docker tag "${BASE_IMAGE}" "${FINAL_IMAGE}"

if [ "$SKIP_PUSH" -eq 0 ]; then
  echo "📤 Pushing to NGC: ${FINAL_IMAGE}"
  /home/zhengyuz/ngc-cli/ngc registry image push "${FINAL_IMAGE}"
  echo "📤 Image pushed: ${FINAL_IMAGE}"
else
  echo "⏭️  Skipping push (--skip-push)"
fi

# =============================================================================
# Auto-cleanup: Remove old deps images after successful build
# Policy:
#   - Track TAG → DEPS_HASH mappings in state file
#   - Keep the most recent deps image for each TAG
#   - Remove mappings/images older than 30 days
# =============================================================================
echo ""
echo "🧹 Cleaning up old deps images..."

STATE_FILE="${HOME}/.isaaclab-deps-cache.txt"
MAX_AGE_DAYS=30
old_deps_removed=0

# Update state file: TAG|DEPS_HASH|TIMESTAMP
touch "$STATE_FILE"
# Remove old entry for this tag, add new one
grep -v "^${TAG}|" "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null || true
echo "${TAG}|${DEPS_HASH}|$(date +%s)" >> "${STATE_FILE}.tmp"
mv "${STATE_FILE}.tmp" "$STATE_FILE"

# Build list of deps hashes to keep (from state file)
keep_hashes=""
now_epoch=$(date +%s)

# Read state file, remove old entries, collect hashes to keep
> "${STATE_FILE}.tmp"
if [ -f "$STATE_FILE" ] && [ -s "$STATE_FILE" ]; then
  while IFS='|' read -r tag hash timestamp; do
    [ -z "$tag" ] && continue
    [ -z "$hash" ] && continue
    [ -z "$timestamp" ] && continue
    
    age_days=$(( (now_epoch - timestamp) / 86400 ))
    
    if [ "$age_days" -le "$MAX_AGE_DAYS" ]; then
      # Keep this mapping
      echo "${tag}|${hash}|${timestamp}" >> "${STATE_FILE}.tmp"
      keep_hashes="${keep_hashes} isaac-lab-deps:${hash}"
      echo "   ✓ Tag '${tag}' → deps:${hash} (${age_days}d old)"
    else
      echo "   ⏰ Tag '${tag}' mapping expired (${age_days}d old)"
    fi
  done < "$STATE_FILE"
fi
if [ -f "${STATE_FILE}.tmp" ]; then
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
fi

# Remove deps images not in keep list
deps_images=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^isaac-lab-deps:" || true)

for img in $deps_images; do
  if echo "$keep_hashes" | grep -q "$img"; then
    : # Already logged above
  else
    echo "   🗑 Removing: $img (not mapped to any tag)"
    docker rmi "$img" 2>/dev/null && ((old_deps_removed++)) || echo "      (in use, skipped)"
  fi
done

if [ "$old_deps_removed" -gt 0 ]; then
  echo "   Removed $old_deps_removed old deps image(s)"
fi

# Prune dangling images silently
docker image prune -f > /dev/null 2>&1 || true

echo ""
echo "✅ Done!"
