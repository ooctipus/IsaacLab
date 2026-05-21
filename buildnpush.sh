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
#   ./buildnpush.sh tag <source-image> <tag> [--skip-push]
#   ./buildnpush.sh enter <tag>       # Enter a container for the given tag
#   ./buildnpush.sh --status          # Show images and disk usage
#   ./buildnpush.sh --clean           # Remove old deps images (keep current)
#   ./buildnpush.sh --clean-all       # Remove ALL Isaac Lab images
#
# Build Options (ordered by depth of rebuild, light to heavy):
#   -s, --source      Rebuild source layer only (no pip)
#   -p, --pip         Rebuild source + re-run pip install
#   -d, --deps        Rebuild deps + source + pip (uses Docker cache)
#   -a, --all         Full rebuild from scratch (no Docker cache)
#       --skip-push   Build/tag only, don't push to NGC
#   -h, --help        Show this help message
#
# Examples:
#   ./buildnpush.sh v1.0          # Smart build (skip deps if unchanged)
#   ./buildnpush.sh v1.0 -s       # Source-only rebuild
#   ./buildnpush.sh v1.0 -p       # Source + pip rebuild (use after setup.py edits)
#   ./buildnpush.sh v1.0 -d       # Deps rebuild
#   ./buildnpush.sh v1.0 -a       # Full rebuild from scratch
#   ./buildnpush.sh tag oocti/isaaclab-manipulation:latest factory
#   ./buildnpush.sh enter v1.0        # Enter container for v1.0
#   ./buildnpush.sh --status          # Check current images
#   ./buildnpush.sh --clean           # Cleanup old deps caches
# =============================================================================

show_help() {
  sed -n "/^# Usage:/,/^# =============================================================================/p" "$0" | sed "s/^# \{0,1\}//"
  exit 0
}

image_exists() {
  docker image inspect "$1" &>/dev/null
}

tag_image() {
  local source_image="${1:-}"
  local tag="${2:-}"
  shift 2 2>/dev/null || true

  local skip_push=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-push)
        skip_push=1
        shift
        ;;
      -h|--help)
        echo "Usage: ./buildnpush.sh tag <source-image> <tag> [--skip-push]"
        exit 0
        ;;
      *)
        echo "Unexpected argument for tag: $1"
        echo "Usage: ./buildnpush.sh tag <source-image> <tag> [--skip-push]"
        exit 1
        ;;
    esac
  done

  if [ -z "$source_image" ] || [ -z "$tag" ]; then
    echo "Error: source image and tag are required."
    echo "Usage: ./buildnpush.sh tag <source-image> <tag> [--skip-push]"
    exit 1
  fi

  if ! image_exists "$source_image"; then
    echo "Error: source image not found locally: ${source_image}"
    echo "Pull it first, e.g.: docker pull ${source_image}"
    exit 1
  fi

  local final_image="nvcr.io/nvidian/octi-isaac-lab:${tag}"
  echo "🏷  Retagging image"
  echo "   Source: ${source_image}"
  echo "   Target: ${final_image}"
  docker tag "$source_image" "$final_image"

  if [ "$skip_push" -eq 0 ]; then
    echo "📤 Pushing to NGC: ${final_image}"
    /home/zhengyuz/ngc-cli/ngc registry image push "$final_image"
    echo "📤 Image pushed: ${final_image}"
  else
    echo "⏭️  Skipping push (--skip-push)"
  fi

  echo "✅ Done!"
  exit 0
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
REBUILD_ALL=0
REBUILD_DEPS=0
REBUILD_PIP=0
REBUILD_SOURCE=0
SKIP_PUSH=0
TAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    tag)
      shift
      tag_image "$@"
      ;;
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
    -a|--all)
      REBUILD_ALL=1
      shift
      ;;
    -d|--deps)
      REBUILD_DEPS=1
      shift
      ;;
    -p|--pip)
      REBUILD_PIP=1
      shift
      ;;
    -s|--source)
      REBUILD_SOURCE=1
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

if [ "$REBUILD_SOURCE" -eq 1 ] && { [ "$REBUILD_ALL" -eq 1 ] || [ "$REBUILD_DEPS" -eq 1 ] || [ "$REBUILD_PIP" -eq 1 ]; }; then
  echo "Error: -s/--source cannot be combined with -a/--all, -d/--deps, or -p/--pip."
  exit 1
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

# Layer-budget gate: deps caches accumulate layers as ``-p/--pip`` builds stack
# source/pip update layers on top. Docker's overlay driver caps an image at 127
# layers; we keep well below that so a budgeted cache always has room for at
# least one more source-only update on top.
#
# Reference points (Dockerfile.base + ``--chown`` Dockerfile.source-only):
#   - Clean ``-d/--deps`` cache: ~71 layers
#   - After 1 ``-p/--pip`` on top: ~80 layers
#   - Each subsequent ``-p/--pip`` adds ~9 layers
# A budget of 110 lets ~4 ``-p/--pip`` rounds stack before the user is nudged
# to flatten with ``-d/--deps``.
MAX_LAYERS_FOR_DEPS_CACHE=110

image_layer_count() {
  # ``docker history`` includes a header line; subtract it to get the layer count.
  docker history --no-trunc "$1" 2>/dev/null | tail -n +2 | wc -l
}

image_within_layer_budget() {
  local count
  count=$(image_layer_count "$1")
  # An image with 0 layers means ``docker history`` failed (e.g. image missing);
  # treat that as out-of-budget so we fall through to the error/rebuild paths.
  [ "$count" -gt 0 ] && [ "$count" -lt "$MAX_LAYERS_FOR_DEPS_CACHE" ]
}

# Pick the most recently created ``isaac-lab-deps:*`` image that still has layer
# headroom. Newest-first ordering means promoted ``-p/--pip`` results (which
# carry the latest pip state) outrank older clean deps caches.
newest_within_budget_deps_image() {
  local img
  for img in $(
    docker images --filter 'reference=isaac-lab-deps:*' \
      --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' 2>/dev/null \
      | sort -k1,1 -r | awk '{print $NF}'
  ); do
    if image_within_layer_budget "$img"; then
      echo "$img"
      return
    fi
  done
}

any_within_budget_deps_image_exists() {
  [ -n "$(newest_within_budget_deps_image)" ]
}

if [ "$REBUILD_SOURCE" -eq 1 ]; then
  SKIP_DEPS=1
  USE_CACHE=1
  if image_exists "$DEPS_IMAGE" && image_within_layer_budget "$DEPS_IMAGE"; then
    echo "   ✓ Cached deps image found: ${DEPS_IMAGE} ($(image_layer_count "$DEPS_IMAGE") layers)"
  elif any_within_budget_deps_image_exists; then
    EXISTING_DEPS=$(newest_within_budget_deps_image)
    echo "   ⚠ Hash mismatch (or layer-budget exceeded on ${DEPS_IMAGE}); using newest in-budget cache: ${EXISTING_DEPS} ($(image_layer_count "$EXISTING_DEPS") layers)"
    DEPS_IMAGE="$EXISTING_DEPS"
  else
    echo "   ✗ No deps image is within ${MAX_LAYERS_FOR_DEPS_CACHE}-layer budget."
    echo "     Re-run without -s/--source or use -d/--deps to flatten and rebuild."
    exit 1
  fi
  REASON="source-only rebuild on cached deps (-s/--source)"
elif [ "$REBUILD_ALL" -eq 1 ]; then
  SKIP_DEPS=0
  USE_CACHE=0
  REASON="full rebuild (-a/--all)"
elif [ "$REBUILD_DEPS" -eq 1 ]; then
  SKIP_DEPS=0
  USE_CACHE=1
  REASON="deps rebuild (-d/--deps)"
elif [ "$REBUILD_PIP" -eq 1 ]; then
  SKIP_DEPS=1
  USE_CACHE=1
  # Use whatever deps image exists (hash may have changed due to setup.py edits).
  # Prefer the current-hash deps cache if it still has layer headroom; otherwise
  # fall back to the newest in-budget cache so we don't keep stacking layers on
  # an already-tall image.
  if image_exists "$DEPS_IMAGE" && image_within_layer_budget "$DEPS_IMAGE"; then
    echo "   ✓ Using current-hash deps cache: ${DEPS_IMAGE} ($(image_layer_count "$DEPS_IMAGE") layers)"
  elif image_exists "$DEPS_IMAGE"; then
    EXISTING_DEPS=$(newest_within_budget_deps_image)
    if [ -z "$EXISTING_DEPS" ]; then
      echo "   ✗ ${DEPS_IMAGE} exceeds ${MAX_LAYERS_FOR_DEPS_CACHE}-layer budget and no other cache fits."
      echo "     Re-run with -d/--deps to flatten and rebuild."
      exit 1
    fi
    echo "   ⚠ ${DEPS_IMAGE} exceeds layer budget ($(image_layer_count "$DEPS_IMAGE") layers); using ${EXISTING_DEPS}"
    DEPS_IMAGE="$EXISTING_DEPS"
  elif any_within_budget_deps_image_exists; then
    EXISTING_DEPS=$(newest_within_budget_deps_image)
    echo "   ⚠ Deps hash changed but -p/--pip: using ${EXISTING_DEPS} ($(image_layer_count "$EXISTING_DEPS") layers)"
    DEPS_IMAGE="$EXISTING_DEPS"
  else
    echo "   ✗ No deps image available for -p/--pip."
    echo "     Re-run without -p/--pip or use -d/--deps."
    exit 1
  fi
  REASON="source + pip rebuild on cached deps (-p/--pip)"
elif image_exists "$DEPS_IMAGE"; then
  if ! image_within_layer_budget "$DEPS_IMAGE"; then
    SKIP_DEPS=0
    USE_CACHE=1
    REASON="cached deps image exceeds layer budget; rebuilding deps"
    echo "   ⚠ ${DEPS_IMAGE} has $(image_layer_count "$DEPS_IMAGE") layers (budget ${MAX_LAYERS_FOR_DEPS_CACHE}) - full rebuild"
  else
    # Exact hash match - use cached deps
    SKIP_DEPS=1
    USE_CACHE=1
    REASON="deps cached (${DEPS_IMAGE})"
    echo "   ✓ Cached deps image found: ${DEPS_IMAGE} ($(image_layer_count "$DEPS_IMAGE") layers)"
  fi
elif ! any_within_budget_deps_image_exists && image_exists "$BASE_IMAGE" && image_within_layer_budget "$BASE_IMAGE"; then
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
  if any_within_budget_deps_image_exists; then
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
    --build-arg RUN_PIP_INSTALL="${REBUILD_PIP}" \
    -t "${BASE_IMAGE}" \
    .

  # Promote a successful ``-p/--pip`` build to the deps cache so that subsequent
  # ``-s/--source`` / ``-p/--pip`` runs see the latest pip state (e.g. an upgraded
  # ``warp-lang`` from a setup.py bump). The layer-budget gate prevents runaway
  # stacking on Docker's overlay driver — when the budget is exceeded the user
  # is told to flatten with ``-d/--deps``.
  if [ "$REBUILD_PIP" -eq 1 ]; then
    new_layer_count=$(image_layer_count "${BASE_IMAGE}")
    if [ "${new_layer_count}" -lt "${MAX_LAYERS_FOR_DEPS_CACHE}" ]; then
      echo "🏷  Promoting -p/--pip build to deps cache: ${DEPS_IMAGE} (${new_layer_count}/${MAX_LAYERS_FOR_DEPS_CACHE} layers)"
      docker tag "${BASE_IMAGE}" "${DEPS_IMAGE}"
    else
      echo "⚠ Layer count ${new_layer_count} ≥ ${MAX_LAYERS_FOR_DEPS_CACHE}; skipping deps cache promotion."
      echo "  Run with -d/--deps to flatten and reset the layer count."
    fi
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
