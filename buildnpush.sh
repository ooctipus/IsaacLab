#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build and Push Script for Isaac Lab Docker Images
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
#   --force-pip-only  Re-run pip install on cached deps (no isaacsim rebuild)
#   --skip-push       Build only, don't push to NGC
#   -h, --help        Show this help message
#
# Examples:
#   ./buildnpush.sh v1.0              # Smart build (skip deps if unchanged)
#   ./buildnpush.sh v1.0 --force      # Full rebuild
#   ./buildnpush.sh v1.0 --force-pip-only  # Re-run pip install only
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

  local docker_args=(
    --gpus all
    --network host
    -e "ACCEPT_EULA=Y"
    -e "OMNI_KIT_ALLOW_ROOT=1"
    -e "WANDB_USERNAME=${WANDB_USERNAME:-nvidia}"
  )
  if [ -n "${WANDB_API_KEY:-}" ]; then
    docker_args+=(-e "WANDB_API_KEY=${WANDB_API_KEY}")
  fi
  if [ -f "${HOME}/.netrc" ]; then
    docker_args+=(-v "${HOME}/.netrc:/root/.netrc:ro")
  fi

  add_cache_mount() {
    local env_name="$1"
    local default_source="$2"
    local shared_root_suffix="$3"
    local target="$4"
    local source="${!env_name:-}"

    if [ -z "$source" ] && [ -n "${HOST_ISAACLAB_CACHE_ROOT:-}" ]; then
      source="${HOST_ISAACLAB_CACHE_ROOT}/${shared_root_suffix}"
    fi
    if [ -z "$source" ]; then
      source="$default_source"
    fi
    if [ -d "$source" ]; then
      docker_args+=(-v "$(readlink -f "$source"):${target}:rw")
    fi
  }

  add_cache_mount HOST_ISAACSIM_KIT_CACHE_PATH "_isaac_sim/kit/cache" "isaac-sim/kit/cache" "/isaac-sim/kit/cache"
  add_cache_mount HOST_OMNIVERSE_CACHE_PATH "${HOME}/.cache/ov" "ov" "/root/.cache/ov"
  add_cache_mount HOST_NVIDIA_GL_CACHE_PATH \
    "${HOME}/.cache/nvidia/GLCache" "nvidia/GLCache" "/root/.cache/nvidia/GLCache"
  add_cache_mount HOST_NVIDIA_COMPUTE_CACHE_PATH "${HOME}/.nv/ComputeCache" "nv/ComputeCache" "/root/.nv/ComputeCache"
  add_cache_mount HOST_NVIDIA_OPTIX_CACHE_PATH \
    "${HOME}/.cache/NVIDIA/OptixCache" "NVIDIA/OptixCache" "/root/.cache/NVIDIA/OptixCache"

  mkdir -p models_tmp
  docker_args+=(-v "$(pwd)/models_tmp:/workspace/isaaclab/models_tmp:rw")

  if [ "$mount_local" = "--mount" ]; then
    docker_args+=(-v "$(pwd):/local:rw")
    echo "🚀 Entering container: ${image} (local mounted at /local)"
  else
    echo "🚀 Entering container: ${image}"
  fi

  docker run -it --rm \
    "${docker_args[@]}" \
    --entrypoint /bin/bash \
    "${image}"
  exit 0
}

# =============================================================================
# Cleanup and Status Commands
# =============================================================================
show_status() {
  echo "📊 Isaac Lab Docker Images Status"
  echo "=================================="
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
  # Hash files that affect the dependency/base image. This includes the Isaac Sim
  # base image selection, Docker build recipe, package metadata, and extension apt deps.
  {
    for file in docker/.env.base docker/Dockerfile.base docker/docker-compose.yaml pyproject.toml environment.yml isaaclab.sh; do
      if [ -f "$file" ]; then
        echo "### ${file}"
        cat "$file"
      fi
    done

    find source \( -name "setup.py" -o -name "requirements.txt" -o -name "extension.toml" \) -type f 2>/dev/null | \
      sort | \
      while IFS= read -r file; do
        echo "### ${file}"
        cat "$file"
      done
  } | md5sum | cut -d' ' -f1 | head -c 12
}

# =============================================================================
# Parse Arguments
# =============================================================================
FORCE_REBUILD=0
FORCE_DEPS=0
FORCE_PIP=0
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
    --force-pip-only)
      FORCE_PIP=1
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
DEPS_IMAGE="isaac-lab-deps:${DEPS_HASH}"
BASE_IMAGE="isaac-lab-base"
FINAL_IMAGE="nvcr.io/nvidian/octi-isaac-lab:${TAG}"

echo "🔍 Dependency Analysis..."
echo "   Hash: ${DEPS_HASH}"

# =============================================================================
# Determine Build Strategy
# =============================================================================
SKIP_DEPS=0
USE_CACHE=1
REASON=""

# Check if ANY isaac-lab-deps:* images exist (indicates we're already using new system)
is_source_only_layered_image() {
  docker history --no-trunc "$1" 2>/dev/null | grep -Eq "COPY (\\.\\./)?source( |$)"
}

first_clean_deps_image() {
  local img
  for img in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^isaac-lab-deps:" || true); do
    if ! is_source_only_layered_image "$img"; then
      echo "$img"
      return
    fi
  done
}

any_clean_deps_image_exists() {
  [ -n "$(first_clean_deps_image)" ]
}

if [ "$FORCE_REBUILD" -eq 1 ]; then
  SKIP_DEPS=0
  USE_CACHE=0
  REASON="forced full rebuild (--force)"
elif [ "$FORCE_DEPS" -eq 1 ]; then
  SKIP_DEPS=0
  USE_CACHE=1
  REASON="forced deps rebuild (--force-deps)"
elif [ "$FORCE_PIP" -eq 1 ]; then
  SKIP_DEPS=1
  USE_CACHE=1
  # Use whatever deps image exists (hash may have changed due to setup.py edits)
  if image_exists "$DEPS_IMAGE" && is_source_only_layered_image "$DEPS_IMAGE"; then
    EXISTING_DEPS=$(first_clean_deps_image)
    if [ -z "$EXISTING_DEPS" ]; then
      echo "   ✗ ${DEPS_IMAGE} is source-only layered and no clean deps cache exists."
      echo "     Re-run with --force-deps to rebuild a clean deps image."
      exit 1
    fi
    echo "   ⚠ ${DEPS_IMAGE} is source-only layered; using ${EXISTING_DEPS}"
    DEPS_IMAGE="$EXISTING_DEPS"
  elif ! image_exists "$DEPS_IMAGE" && any_clean_deps_image_exists; then
    EXISTING_DEPS=$(first_clean_deps_image)
    echo "   ⚠ Deps hash changed but --force-pip-only: using ${EXISTING_DEPS}"
    DEPS_IMAGE="$EXISTING_DEPS"
  elif ! image_exists "$DEPS_IMAGE"; then
    echo "   ✗ No deps image available for --force-pip-only."
    echo "     Re-run without --force-pip-only or use --force-deps."
    exit 1
  fi
  REASON="pip-only rebuild on existing deps (--force-pip-only)"
elif image_exists "$DEPS_IMAGE"; then
  if is_source_only_layered_image "$DEPS_IMAGE"; then
    SKIP_DEPS=0
    USE_CACHE=1
    REASON="cached deps image is source-only layered; rebuilding deps"
    echo "   ⚠ ${DEPS_IMAGE} contains source-only layers - full rebuild required"
  else
    # Exact hash match - use cached deps
    SKIP_DEPS=1
    USE_CACHE=1
    REASON="deps cached (${DEPS_IMAGE})"
    echo "   ✓ Cached deps image found: ${DEPS_IMAGE}"
  fi
elif ! any_clean_deps_image_exists && image_exists "$BASE_IMAGE" && ! is_source_only_layered_image "$BASE_IMAGE"; then
  # ONE-TIME MIGRATION: No deps images exist yet, but isaac-lab-base exists
  # This means we're migrating from old system to new system
  echo "   ⚡ One-time migration: tagging existing ${BASE_IMAGE} as deps cache..."
  docker tag "${BASE_IMAGE}" "${DEPS_IMAGE}"
  SKIP_DEPS=1
  USE_CACHE=1
  REASON="migrated existing image (one-time)"
  echo "   ✓ Tagged as ${DEPS_IMAGE}"
else
  # Deps changed (hash doesn't match any existing deps image)
  SKIP_DEPS=0
  USE_CACHE=1
  if any_clean_deps_image_exists; then
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
echo "   Strategy:      ${REASON}"
echo "   Pip Install:   $([ "$SKIP_DEPS" -eq 0 ] && echo "YES" || echo "SKIP (cached)")"
echo "   Docker Cache:  $([ "$USE_CACHE" -eq 1 ] && echo "YES" || echo "NO")"
echo "   Push to NGC:   $([ "$SKIP_PUSH" -eq 0 ] && echo "YES" || echo "SKIP")"
echo ""

# =============================================================================
# Symlink Resolution (Docker COPY doesn't follow external symlinks)
# =============================================================================
ASSETS_DATA_DIR="source/isaaclab_assets/data"
# Top-level symlinks (relative to repo root) that the Dockerfile copies in.
# Each must be resolved to its target before `docker build` and restored after.
EXTRA_SYMLINKS=(
  "dep"
  "dep/rsl_rl"
)
SYMLINKS_FILE=$(mktemp)

_resolve_one_symlink() {
  local item="$1"
  if [ ! -L "$item" ]; then
    return
  fi
  local target
  target=$(readlink -f "$item")
  if [ ! -d "$target" ]; then
    return
  fi
  echo "   📁 Resolving: $item -> $target"
  rm "$item"
  cp -rL "$target" "$item"
  echo "$item:$target" >> "$SYMLINKS_FILE"
}

resolve_symlinks() {
  echo "🔗 Resolving symlinks for build context..."
  if [ -d "${ASSETS_DATA_DIR}" ]; then
    for item in "${ASSETS_DATA_DIR}"/*; do
      _resolve_one_symlink "$item"
    done
  fi
  for item in "${EXTRA_SYMLINKS[@]}"; do
    _resolve_one_symlink "$item"
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
  export SKIP_PIP_INSTALL=0
  if [ "$USE_CACHE" -eq 0 ]; then
    export ISAACLAB_NOCACHE=1
  else
    unset ISAACLAB_NOCACHE 2>/dev/null || true
  fi

  ./docker/container.py start --build

  # Tag as deps image for future cache hits
  echo "🏷  Caching deps image as ${DEPS_IMAGE}"
  docker tag "${BASE_IMAGE}" "${DEPS_IMAGE}"
else
  echo "▶️  Using cached dependencies, copying source only..."

  # Load env vars from .env.base
  source docker/.env.base

  # Build using lightweight source-only Dockerfile
  docker build \
    --no-cache \
    -f docker/Dockerfile.source-only \
    --build-arg DEPS_BASE_IMAGE="${DEPS_IMAGE}" \
    --build-arg ISAACLAB_PATH_ARG="${DOCKER_ISAACLAB_PATH}" \
    --build-arg ISAACSIM_ROOT_PATH_ARG="${DOCKER_ISAACSIM_ROOT_PATH}" \
    --build-arg RUN_PIP_INSTALL="${FORCE_PIP}" \
    -t "${BASE_IMAGE}" \
    .

  # Do not tag source-only images as deps cache. Re-tagging here stacks
  # source-copy layers on future source-only builds and eventually trips
  # Docker overlay's max-depth limit. Use --force-deps for a clean deps cache.
  if [ "$FORCE_PIP" -eq 1 ]; then
    echo "ℹ️  Not updating deps cache from a source-only build. Use --force-deps for a clean deps cache."
  fi
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
while IFS='|' read -r tag hash timestamp; do
  [ -z "$tag" ] && continue

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
mv "${STATE_FILE}.tmp" "$STATE_FILE"

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
