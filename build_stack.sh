#!/usr/bin/env bash
set -Eeuo pipefail
trap 'err "Failed at line $LINENO (exit $?)"; exit 1' ERR
# ============================================================
# Omniverse Stack Builder (PhysX, Kit, Isaac Sim, Isaac Lab)
# ============================================================

# ---------- logging ----------
validate_bool() {
  case "${1,,}" in
    true|false|1|0|yes|no|on|off) ;;
    *) die "Invalid boolean for $2: '$1' (use true/false)";;
  esac
}

normalize_bool() {
  # Echo canonical true/false
  if truthy "$1"; then echo "true"
  elif falsy "$1"; then echo "false"
  else die "Invalid boolean value: '$1'"
  fi
}

if [[ -t 1 ]]; then
  c_bold=$(tput bold); c_reset=$(tput sgr0)
  c_dim=$(tput dim)
  c_red=$(tput setaf 1); c_green=$(tput setaf 2); c_yellow=$(tput setaf 3); c_blue=$(tput setaf 4)
else
  c_bold=""; c_reset=""; c_dim=""; c_red=""; c_green=""; c_yellow=""; c_blue=""
fi
log()    { echo -e "${c_dim}[$(date +%H:%M:%S)]${c_reset} $*"; }
info()   { echo -e "${c_blue}${c_bold}INFO${c_reset}  $*"; }
ok()     { echo -e "${c_green}${c_bold}OK${c_reset}    $*"; }
warn()   { echo -e "${c_yellow}${c_bold}WARN${c_reset}  $*"; }
err()    { echo -e "${c_red}${c_bold}ERROR${c_reset} $*"; }
die()    { err "$*"; exit 1; }

truthy() { case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }
falsy()  { case "${1,,}" in 0|false|no|off) return 0;; *) return 1;; esac; }

# ---------- defaults ----------
ROOT="../omniverse-stack-$(date +%Y%m%d%H%M)"
JOBS="${JOBS:-8}"
BUILD_CONFIG="release"          # release|debug
DRY_RUN="false"

# Repos & defaults
PHYSX_NAME="physx";   PHYSX_DIR=""
PHYSX_SSH="ssh://git@gitlab-master.nvidia.com:12051/omniverse/physics.git"
PHYSX_BRANCH="trunk"; PHYSX_COMMIT=""
PHYSX_BUILD="false"   # opt-in

KIT_NAME="kit";       KIT_DIR=""
KIT_SSH="ssh://git@gitlab-master.nvidia.com:12051/omniverse/rtxdev/kit.git"
KIT_BRANCH="master";
# Default pin for Kit commit (used only if user doesn't specify a branch)
KIT_COMMIT_DEFAULT="021092213040248918852c687bf0a8120a6e48d0"
KIT_COMMIT="$KIT_COMMIT_DEFAULT"
KIT_BUILD="false"     # opt-in

SIM_NAME="sim";       SIM_DIR=""
SIM_SSH="ssh://git@gitlab-master.nvidia.com:12051/omniverse/isaac/omni_isaac_sim.git"
SIM_BRANCH="release/5.0_oss";
# Default pin for Sim commit (used only if user doesn't specify a branch)
SIM_COMMIT_DEFAULT="9fd50328d0e0ebfa79eb591e872eed0ebb68eaf5"
SIM_COMMIT="$SIM_COMMIT_DEFAULT"
SIM_BUILD="true"

LAB_NAME="lab";       LAB_DIR=""
LAB_SSH="git@github.com:isaac-sim/IsaacLab.git"
LAB_BRANCH="main";    LAB_COMMIT=""
LAB_BUILD="true"
# Recreate Lab conda env if it exists
LAB_RECREATE="false"

# Custom link flags (Sim -> custom Kit/PhysX; Lab -> custom PhysX exts)
# USE_CUSTOM_KIT auto => true if Kit step/ref/build requested
USE_CUSTOM_KIT="auto"     # auto|true|false
USE_CUSTOM_PHYSX="auto"   # auto|true|false (auto => true iff PHYSX_BUILD=true)
LAB_USE_PHYSX_EXTS="false"

# Conda env name for Isaac Lab (if not provided, derive from root folder)
LAB_ENV_NAME=""

# Step selection and git behavior
STEPS="physx,kit,sim,lab"   # comma-separated; e.g., "lab" or "sim,lab"
GIT_UPDATE="true"           # false => skip fetch/checkout/reset; requires repos exist

# Confirmation flags
ASSUME_YES="false"
ASSUME_NO="false"

# Track user overrides (auto-enable build if a ref is supplied)
PHYSX_REF_USERSET="false"
KIT_REF_USERSET="false"
# Track whether user explicitly set commit refs (so branch won't clear them)
KIT_COMMIT_USERSET="false"
SIM_COMMIT_USERSET="false"
PHYSX_BUILD_USERSET=""
KIT_BUILD_USERSET=""

# ---------- helpers ----------
platform_target() {
  local os=$(uname -s) arch=$(uname -m)
  case "$os-$arch" in
    Linux-x86_64) echo "linux-x86_64" ;;
    Linux-aarch64) echo "linux-aarch64" ;;
    Darwin-arm64) echo "macos-arm64"  ;;
    Darwin-x86_64) echo "macos-x86_64" ;;
    *) echo ""; return 1 ;;
  esac
}
run() { if truthy "$DRY_RUN"; then echo "+ $*"; else eval "$@"; fi; }
ensure_tool() { command -v "$1" >/dev/null 2>&1 || die "Missing required tool: $1"; }
in_steps() {
  local name="$1"
  [[ ",${STEPS}," == *",${name},"* ]]
}

# Miniconda bootstrap (prompt to install if missing)
install_miniconda() {
  local os=$(uname -s) arch=$(uname -m) fn url
  case "$os-$arch" in
    Linux-x86_64)  fn="Miniconda3-latest-Linux-x86_64.sh" ;;
    Linux-aarch64) fn="Miniconda3-latest-Linux-aarch64.sh" ;;
    Darwin-arm64)  fn="Miniconda3-latest-MacOSX-arm64.sh" ;;
    Darwin-x86_64) fn="Miniconda3-latest-MacOSX-x86_64.sh" ;;
    *) die "Unsupported platform for Miniconda auto-install: $os-$arch" ;;
  esac
  url="https://repo.anaconda.com/miniconda/$fn"
  local dest="${TMPDIR:-/tmp}/$fn"

  info "Downloading Miniconda3 installer ($fn)"
  if command -v curl >/dev/null 2>&1; then
    run "curl -fsSL \"$url\" -o \"$dest\""
  elif command -v wget >/dev/null 2>&1; then
    run "wget -q \"$url\" -O \"$dest\""
  else
    die "Neither curl nor wget found to download Miniconda."
  fi

  info "Installing Miniconda3 to \$HOME/miniconda3"
  run "bash \"$dest\" -b -p \"$HOME/miniconda3\""
  local conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
  [[ -f "$conda_sh" ]] || die "conda.sh not found at $conda_sh"

  # Make conda available in this shell
  if truthy "$DRY_RUN"; then
    echo "+ source \"$conda_sh\""
  else
    # shellcheck disable=SC1090
    source "$conda_sh"
  fi
  ok "Miniconda installed"
  run "conda --version"
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  warn "Conda not found."
  local resp=""
  if truthy "$ASSUME_NO"; then
    die "Conda is required for the Lab step. Re-run with --steps without 'lab' or allow install."
  fi
  if truthy "$ASSUME_YES"; then
    resp="y"
  else
    if [[ -t 0 ]]; then
      read -r -p "Install Miniconda3 locally at \$HOME/miniconda3 now? [y/N]: " resp
    else
      die "Conda not found and no TTY to confirm install. Re-run with -y to auto-install, or skip the Lab step."
    fi
  fi
  case "${resp,,}" in
    y|yes) install_miniconda ;;
    *) die "Conda is required for the Lab step. Aborting." ;;
  esac
}

# Map dotted keys and flags to variables
set_kv() {
  local k="$1" v="$2"
  case "$k" in
    -y|--yes) ASSUME_YES="true" ;;
    -n|--no)  ASSUME_NO="true" ;;
    root|--root) ROOT="$v" ;;
    --jobs) JOBS="$v" ;;
    --config) BUILD_CONFIG="$v" ;;
    --dry-run) DRY_RUN="$v" ;;

    steps|--steps) STEPS="$v" ;;
    --git-update)  GIT_UPDATE="$v" ;;

    physx.ssh|--physx-ssh) PHYSX_SSH="$v" ;;
    physx.branch|--physx-branch) PHYSX_BRANCH="$v"; PHYSX_REF_USERSET="true" ;;
    physx.commit|--physx-commit) PHYSX_COMMIT="$v"; PHYSX_REF_USERSET="true" ;;
    physx.build|--physx-build)   PHYSX_BUILD="$v";  PHYSX_BUILD_USERSET="true" ;;

    kit.ssh|--kit-ssh) KIT_SSH="$v" ;;
    kit.branch|--kit-branch)
      KIT_BRANCH="$v"; KIT_REF_USERSET="true"
      # If user sets a branch and commit is still the default pin (not explicitly set), clear it
      if [[ "$KIT_COMMIT_USERSET" != "true" && "$KIT_COMMIT" == "$KIT_COMMIT_DEFAULT" ]]; then
        KIT_COMMIT=""
      fi
      ;;
    kit.commit|--kit-commit)
      KIT_COMMIT="$v"; KIT_REF_USERSET="true"; KIT_COMMIT_USERSET="true"
      ;;
    kit.build|--kit-build)       KIT_BUILD="$v";    KIT_BUILD_USERSET="true" ;;

    sim.ssh|--sim-ssh) SIM_SSH="$v" ;;
    sim.branch|--sim-branch)
      SIM_BRANCH="$v"
      # If user sets a branch and commit is still the default pin (not explicitly set), clear it
      if [[ "$SIM_COMMIT_USERSET" != "true" && "$SIM_COMMIT" == "$SIM_COMMIT_DEFAULT" ]]; then
        SIM_COMMIT=""
      fi
      ;;
    sim.commit|--sim-commit)
      SIM_COMMIT="$v"; SIM_COMMIT_USERSET="true"
      ;;
    sim.build|--sim-build)   SIM_BUILD="$v" ;;

    lab.ssh|--lab-ssh) LAB_SSH="$v" ;;
    lab.branch|--lab-branch) LAB_BRANCH="$v" ;;
    lab.commit|--lab-commit) LAB_COMMIT="$v" ;;
    lab.build|--lab-build)   LAB_BUILD="$v" ;;
    lab.env|--lab-env)       LAB_ENV_NAME="$v" ;;
    lab.recreate|--lab-recreate) LAB_RECREATE="$v" ;;

    use.custom.kit|--use-custom-kit) USE_CUSTOM_KIT="$v" ;;
    use.custom.physx|--use-custom-physx) USE_CUSTOM_PHYSX="$v" ;;
    lab.use.physx.exts|--lab-use-physx-exts) LAB_USE_PHYSX_EXTS="$v" ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $k";;
  esac
}

usage() {
  cat <<EOF
${c_bold}Omniverse Stack Builder${c_reset}

${c_bold}Usage:${c_reset}
  $(basename "$0") [options] [key=value ...]

${c_bold}Common options:${c_reset}
  -y, --yes                   Proceed without interactive confirmation (also auto-installs Miniconda if missing)
  -n, --no                    Cancel without running anything
  --root DIR                  Parent folder (default: $ROOT)
  --jobs N                    Parallel jobs (default: $JOBS)
  --config {release|debug}    Build config (default: $BUILD_CONFIG)
  --dry-run {true|false}      Print commands without executing (default: $DRY_RUN)
  --steps LIST                Steps to run (default: $STEPS)
                              e.g., --steps=lab  or  --steps=sim,lab
  --git-update {true|false}   Skip fetch/checkout/reset when false (default: $GIT_UPDATE)

${c_bold}PhysX:${c_reset}
  --physx-ssh URL            (default: $PHYSX_SSH)
  --physx-branch NAME        (default: $PHYSX_BRANCH)
  --physx-commit SHA
  --physx-build {true|false} (default: $PHYSX_BUILD)   # opt-in

${c_bold}Kit:${c_reset}
  --kit-ssh URL              (default: $KIT_SSH)
  --kit-branch NAME          (default: $KIT_BRANCH)
  --kit-commit SHA           (default: $KIT_COMMIT)
  --kit-build {true|false}   (default: $KIT_BUILD)      # opt-in

${c_bold}Isaac Sim:${c_reset}
  --sim-ssh URL              (default: $SIM_SSH)
  --sim-branch NAME          (default: $SIM_BRANCH)
  --sim-commit SHA           (default: $SIM_COMMIT)
  --sim-build {true|false}   (default: $SIM_BUILD)
  --use-custom-kit {auto|true|false}   (default: $USE_CUSTOM_KIT; auto => true iff KIT_BUILD=true)
  --use-custom-physx {auto|true|false} (default: $USE_CUSTOM_PHYSX; auto => true iff PHYSX_BUILD=true)

${c_bold}Isaac Lab:${c_reset}
  --lab-ssh URL              (default: $LAB_SSH)
  --lab-branch NAME          (default: $LAB_BRANCH)
  --lab-commit SHA
  --lab-build {true|false}   (default: $LAB_BUILD)
  --lab-env NAME             (default: derived from root folder)
  --lab-use-physx-exts {true|false} (default: $LAB_USE_PHYSX_EXTS)
  --lab-recreate {true|false} Recreate env if it exists (default: false)

${c_bold}Also accepted:${c_reset} dotted keys like physx.commit=..., sim.branch=..., use.custom.kit=true

${c_bold}Examples:${c_reset}
  $(basename "$0") --root ../stack -y
  $(basename "$0") --root ../stack --kit-commit=0210922... --use-custom-kit=auto -y
  $(basename "$0") --root ../stack --steps=lab --git-update=false -y
  $(basename "$0") --root ../stack --config debug --jobs 16 --dry-run true
EOF
}

# ---------- args ----------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
while (( "$#" )); do
  case "$1" in
    --*=*) k="${1%%=*}"; v="${1#*=}"; set_kv "$k" "$v"; shift;;
    --*)   k="$1"; v="${2:-}"; if [[ -z "$v" || "$v" == --* ]]; then set_kv "$k" ""; shift; else set_kv "$k" "$v"; shift 2; fi;;
    *=*)   k="${1%%=*}"; v="${1#*=}"; set_kv "$k" "$v"; shift;;
    *)     die "Unrecognized argument: $1";;
  esac
done

# ---------- validate & normalize booleans (after args parsed) ----------
validate_bool "$DRY_RUN"         "--dry-run"
validate_bool "$GIT_UPDATE"      "--git-update"
validate_bool "$PHYSX_BUILD"     "--physx-build"
validate_bool "$KIT_BUILD"       "--kit-build"
validate_bool "$SIM_BUILD"       "--sim-build"
validate_bool "$LAB_BUILD"       "--lab-build"
validate_bool "$LAB_RECREATE"    "--lab-recreate"

# allow auto/true/false for custom linkage flags
case "${USE_CUSTOM_KIT,,}" in
  auto|true|false|1|0|yes|no|on|off) ;;
  *) die "Invalid value for --use-custom-kit: '$USE_CUSTOM_KIT' (use auto|true|false)";;
esac
case "${USE_CUSTOM_PHYSX,,}" in
  auto|true|false|1|0|yes|no|on|off) ;;
  *) die "Invalid value for --use-custom-physx: '$USE_CUSTOM_PHYSX' (use auto|true|false)";;
esac

# canonicalize to true/false so later truthy/falsy checks are stable
DRY_RUN=$(normalize_bool "$DRY_RUN")
GIT_UPDATE=$(normalize_bool "$GIT_UPDATE")
PHYSX_BUILD=$(normalize_bool "$PHYSX_BUILD")
KIT_BUILD=$(normalize_bool "$KIT_BUILD")
SIM_BUILD=$(normalize_bool "$SIM_BUILD")
LAB_BUILD=$(normalize_bool "$LAB_BUILD")
LAB_RECREATE=$(normalize_bool "$LAB_RECREATE")

[[ "${USE_CUSTOM_KIT,,}"   != "auto" ]] && USE_CUSTOM_KIT=$(normalize_bool "$USE_CUSTOM_KIT")
[[ "${USE_CUSTOM_PHYSX,,}" != "auto" ]] && USE_CUSTOM_PHYSX=$(normalize_bool "$USE_CUSTOM_PHYSX")

# ---------- sanity ----------
ensure_tool git
platform="$(platform_target || true)" || warn "Unknown platform; Sim/Lab path patching may fail."
[[ -z "$platform" ]] && platform="linux-x86_64"
[[ "$BUILD_CONFIG" =~ ^(release|debug)$ ]] || die "--config must be release or debug"

# Auto-enable build if user passed a ref but not an explicit build toggle
if [[ "$PHYSX_REF_USERSET" == "true" && "$PHYSX_BUILD_USERSET" != "true" ]]; then PHYSX_BUILD="true"; fi
if [[ "$KIT_REF_USERSET"   == "true" && "$KIT_BUILD_USERSET"   != "true" ]]; then KIT_BUILD="true";   fi

# Auto flags (linkage)
# For Kit: enable custom link when user selected kit step, provided kit ref, or enabled kit build
if [[ "$USE_CUSTOM_KIT" == "auto" ]]; then
  if in_steps kit || [[ "$KIT_REF_USERSET" == "true" ]] || [[ "$KIT_BUILD" == "true" ]]; then
    USE_CUSTOM_KIT="true"
  else
    USE_CUSTOM_KIT="false"
  fi
fi
if [[ "$USE_CUSTOM_PHYSX" == "auto" ]]; then USE_CUSTOM_PHYSX="$PHYSX_BUILD"; fi

# ---------- git helpers ----------
clone_or_prepare() {
  local repo_ssh="$1" dest="$2"
  if [[ -d "$dest/.git" ]]; then
    # If existing repo points to a different remote, switch or stop.
    local have_url
    have_url="$(git -C "$dest" remote get-url origin 2>/dev/null || echo "")"
    if [[ -n "$have_url" && "$have_url" != "$repo_ssh" ]]; then
      warn "Repo $dest has origin '$have_url' but you requested '$repo_ssh'."
      if truthy "$ASSUME_YES"; then
        info "Switching origin to '$repo_ssh'."
        run "git -C \"$dest\" remote set-url origin \"$repo_ssh\""
      else
        read -r -p "Switch origin to '$repo_ssh'? [y/N]: " answer
        case "${answer,,}" in
          y|yes) run "git -C \"$dest\" remote set-url origin \"$repo_ssh\"" ;;
          *) die "Remote mismatch. Move '$dest' away or re-run with -y to retarget automatically." ;;
        esac
      fi
    fi
    if truthy "$GIT_UPDATE"; then
      info "Repo exists: $dest — fetching updates"
      run "git -C \"$dest\" fetch --all --tags --prune"
    else
      info "Repo exists: $dest — skipping fetch (git-update=false)"
    fi
  else
    if truthy "$GIT_UPDATE"; then
      info "Cloning into: $dest"
      run "git clone \"$repo_ssh\" \"$dest\""
    else
      die "Repo missing: $dest and --git-update=false (nothing to do)."
    fi
  fi
}
checkout_ref() {
  local dir="$1" branch="$2" commit="$3" name="$4"
  [[ -z "$branch" && -z "$commit" ]] && die "$name: provide either branch or commit"
  (cd "$dir"
    run "git rev-parse --is-inside-work-tree >/dev/null"
    if falsy "$GIT_UPDATE"; then
      info "$name: skipping checkout (git-update=false); HEAD=$(git rev-parse --short HEAD)"
      return 0
    fi
    run "git fetch --all --tags --prune"
    if [[ -n "$branch" && -n "$commit" ]]; then
      warn "$name: both branch ($branch) and commit ($commit) provided; commit takes precedence."
    fi
    if [[ -n "$commit" ]]; then
      info "$name: checking out commit $commit"
      run "git checkout --detach \"$commit\""
    else
      info "$name: checking out branch $branch"
      if git show-ref --verify --quiet "refs/heads/$branch"; then
        run "git checkout \"$branch\""
        run "git reset --hard \"origin/$branch\""
      else
        # Verify the branch exists on origin first
        if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
          run "git checkout -B \"$branch\" \"origin/$branch\""
        else
          die "$name: branch '$branch' not found on origin. Check --${name,,}-ssh and branch name/permissions."
        fi
      fi
    fi
    local head_short
    head_short="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    ok "$name @ $head_short"
  )
}


# ---------- symlink helper ----------
ensure_lab_symlink() {
  local link="$LAB_DIR/_isaac_sim"
  local target="$SIM_DIR/_build/${platform}/${BUILD_CONFIG}"
  [[ -d "$target" ]] || die "Isaac Sim build output not found: $target. Build Sim first or check --config/platform."
  if [[ -L "$link" ]]; then
    local curr; curr="$(readlink "$link")"
    if [[ "$curr" == "$target" ]]; then
      ok "Lab symlink OK: $link -> $target"
    else
      info "Updating Lab symlink: $link -> $target"
      run "ln -sfn \"$target\" \"$link\""
      ok "Lab symlink updated"
    fi
  elif [[ -e "$link" ]]; then
    die "Path exists and is not a symlink: $link. Please remove or rename it."
  else
    info "Creating Lab symlink: $link -> $target"
    run "ln -s \"$target\" \"$link\""
    ok "Lab symlink created"
  fi
}

# ---------- builders ----------
build_physx() {
  [[ "$PHYSX_BUILD" == "true" ]] || { warn "Skipping PhysX build"; return 0; }
  info "Building PhysX (jobs=$JOBS, config=$BUILD_CONFIG)"
  (cd "$PHYSX_DIR"; run "./build.sh -j \"$JOBS\"")
  ok "PhysX build complete"
}
build_kit() {
  [[ "$KIT_BUILD" == "true" ]] || { warn "Skipping Kit build"; return 0; }
  info "Building Kit (repo.sh build)"
  (cd "$KIT_DIR"; run "./repo.sh build")
  ok "Kit build complete"
}
patch_sim_for_custom_kit() {
  local expected="$KIT_DIR/kit/_build/${platform}/${BUILD_CONFIG}"
  local deps="$SIM_DIR/deps"
  local user_out="$deps/kit-sdk.packman.xml.user"
  local kit_build_root="$KIT_DIR/kit/_build/\${platform_target}/\${config}"
  [[ -d "$expected" ]] || warn "Custom Kit build output not found yet: $expected (Sim build may fail until Kit is built)"
  info "Patching Sim to use custom Kit -> $user_out"
  run "mkdir -p \"$deps\""
  cat > "$user_out" <<EOF
<project toolsVersion="6.11" chroniclePath="../_build/chronicles">
  <dependency name="kit_sdk_\${config}" linkPath="../_build/\${platform_target}/\${config}/kit" tags="\${config} non-redist">
    <source path="${kit_build_root}"/>
  </dependency>
</project>
EOF
  ok "Sim now links custom Kit"
}
patch_sim_for_custom_physx() {
  local expected="$PHYSX_DIR/omni/_build/${platform}/${BUILD_CONFIG}"
  local deps="$SIM_DIR/deps"
  local user_out="$deps/omni-physics.packman.xml.user"
  local physx_omni="$PHYSX_DIR/omni"
  [[ -d "$expected" ]] || warn "Custom PhysX build output not found yet: $expected (Sim build may fail until PhysX is built)"
  info "Patching Sim to use custom PhysX -> $user_out"
  run "mkdir -p \"$deps\""
  cat > "$user_out" <<EOF
<project toolsVersion="5.6">
  <dependency name="omni_physics_\${config}" linkPath="../_build/target-deps/omni_physics/\${config}">
    <source path="${physx_omni}" />
  </dependency>
</project>
EOF
  ok "Sim now links custom PhysX"
}
build_sim() {
  [[ "$SIM_BUILD" == "true" ]] || { warn "Skipping Isaac Sim build"; return 0; }
  info "Configuring Sim custom links (kit=$USE_CUSTOM_KIT, physx=$USE_CUSTOM_PHYSX)"
  truthy "$USE_CUSTOM_KIT"   && patch_sim_for_custom_kit || true
  truthy "$USE_CUSTOM_PHYSX" && patch_sim_for_custom_physx || true
  info "Building Isaac Sim (config=$BUILD_CONFIG)"
  (cd "$SIM_DIR"
    if [[ "$BUILD_CONFIG" == "release" ]]; then
      run "./build.sh -r"
    else
      run "./build.sh -d"
    fi
  )
  ok "Isaac Sim build complete"
}
patch_lab_for_physx_exts() {
  local file="$LAB_DIR/source/isaaclab/isaaclab/app/app_launcher.py"
  local exts_path="$PHYSX_DIR/omni/_build/${platform}/${BUILD_CONFIG}/extsPhysics"
  [[ -d "$exts_path" ]] || { warn "PhysX exts path not found: $exts_path"; return 0; }
  info "Patching Isaac Lab app_launcher.py to include PhysX exts devFolders"
  run "cp \"$file\" \"$file.bak\""
  if grep -q 'self\._kit_args *= *\[\]' "$file"; then
    run "sed -i '0,/self\\._kit_args *= *\\[\\]/{s|self\\._kit_args *= *\\[\\]|self._kit_args = [\"--/app/exts/devFolders=[$exts_path]\"]|}' \"$file\""
    ok "Isaac Lab now references PhysX exts"
  else
    warn "Could not find 'self._kit_args = []' in $file; skipping edit (see $file.bak)"
  fi
}
build_lab() {
  [[ "$LAB_BUILD" == "true" ]] || { warn "Skipping Isaac Lab install"; return 0; }
  info "Setting up Isaac Lab (env=$LAB_ENV_NAME)"
  ensure_lab_symlink

  # Ensure conda present (offer to install Miniconda if missing)
  ensure_conda

  # --- Quiet base conda plugin noise (optional) ---
  if conda list -n base conda-anaconda-tos >/dev/null 2>&1; then
    warn "Detected 'conda-anaconda-tos'; updating or removing…"
    run "conda update -n base -c conda-forge conda -y || true"
    # if still present, remove
    if conda list -n base conda-anaconda-tos >/dev/null 2>&1; then
      run "conda remove -n base conda-anaconda-tos -y || true"
    fi
  fi
  # --- end noise fix ---

  (cd "$LAB_DIR"
    local conda_base; conda_base="$(conda info --base 2>/dev/null || true)"
    [[ -n "$conda_base" ]] || die "Conda still not available after attempted install."

    # Create env only if it doesn't exist, or recreate if requested
    local env_exists="false"
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$LAB_ENV_NAME"; then
      env_exists="true"
    fi

    if truthy "$env_exists" && truthy "$LAB_RECREATE"; then
      warn "Conda env '$LAB_ENV_NAME' exists and --lab-recreate=true; removing…"
      run "conda env remove -n \"$LAB_ENV_NAME\" -y || true"
      ok "Removed env '$LAB_ENV_NAME'"
      env_exists="false"
    fi

    if falsy "$env_exists"; then
      info "Creating conda env '$LAB_ENV_NAME' via isaaclab.sh"
      run "./isaaclab.sh -c \"$LAB_ENV_NAME\""
    else
      info "Conda env '$LAB_ENV_NAME' already exists — reusing it."
    fi

    # shellcheck disable=SC1091
    run "set +u; source \"$conda_base/etc/profile.d/conda.sh\" && conda activate \"$LAB_ENV_NAME\" && ./isaaclab.sh -i; set -u"

  )

  truthy "$LAB_USE_PHYSX_EXTS" && patch_lab_for_physx_exts || true
  ok "Isaac Lab setup complete"
}

# ---------- main ----------
info "Root folder: $ROOT"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd -P)"
info "Resolved root: $ROOT"

# If user didn’t set --lab-env, default to the folder name (sanitized)
if [[ -z "$LAB_ENV_NAME" ]]; then
  LAB_ENV_NAME="$(basename "$ROOT")"
  LAB_ENV_NAME="${LAB_ENV_NAME//[^A-Za-z0-9_.-]/_}"   # conda-safe
fi

PHYSX_DIR="$ROOT/$PHYSX_NAME"
KIT_DIR="$ROOT/$KIT_NAME"
SIM_DIR="$ROOT/$SIM_NAME"
LAB_DIR="$ROOT/$LAB_NAME"

# ---------- plan helpers ----------
ref_label() {
  local branch="$1" commit="$2"
  if [[ -n "$commit" ]]; then
    echo "$commit"
  elif [[ -n "$branch" ]]; then
    echo "$branch"
  else
    echo "HEAD"
  fi
}

plan_line() {
  # $1 name, $2 ssh, $3 dest, $4 branch, $5 commit
  local name="$1" ssh="$2" dest="$3" branch="$4" commit="$5"
  echo "  • $name: clone/update -> checkout ($(ref_label "$branch" "$commit"))"
  echo "           - Repo: $ssh"
  echo "           - Dest: $dest"
}

# Plan Summary
info "Plan summary:"
echo "  PhysX:   $PHYSX_SSH  (branch=${PHYSX_BRANCH:-none}, commit=${PHYSX_COMMIT:-none}, build=$PHYSX_BUILD)"
echo "  Kit:     $KIT_SSH     (branch=${KIT_BRANCH:-none},  commit=${KIT_COMMIT:-none},  build=$KIT_BUILD)"
echo "  Sim:     $SIM_SSH     (branch=${SIM_BRANCH:-none},  commit=${SIM_COMMIT:-none},  build=$SIM_BUILD, customKit=$USE_CUSTOM_KIT, customPhysX=$USE_CUSTOM_PHYSX)"
echo "  Lab:     $LAB_SSH     (branch=${LAB_BRANCH:-none},  commit=${LAB_COMMIT:-none},  build=$LAB_BUILD, env=$LAB_ENV_NAME, physxExts=$LAB_USE_PHYSX_EXTS)"
echo "  Config:  $BUILD_CONFIG (platform=$platform), jobs=$JOBS, dry-run=$DRY_RUN"
echo "  Steps:   $STEPS (git-update=$GIT_UPDATE)"
echo

# Execution Plan (respect steps)
echo "${c_bold}Execution plan:${c_reset}"

if in_steps physx; then
  plan_line "PhysX" "$PHYSX_SSH" "$PHYSX_DIR" "$PHYSX_BRANCH" "$PHYSX_COMMIT"
  if truthy "$PHYSX_BUILD"; then
    echo "           - Build: yes (-j $JOBS)"
  else
    echo "           - Build: no"
  fi
else
  echo "  • PhysX: skipped"
fi

if in_steps kit; then
  plan_line "Kit" "$KIT_SSH" "$KIT_DIR" "$KIT_BRANCH" "$KIT_COMMIT"
  if truthy "$KIT_BUILD"; then
    echo "           - Build: yes"
  else
    echo "           - Build: no"
  fi
else
  echo "  • Kit:   skipped"
fi

if in_steps sim; then
  plan_line "Sim" "$SIM_SSH" "$SIM_DIR" "$SIM_BRANCH" "$SIM_COMMIT"
  echo "           - Custom Kit link:   $USE_CUSTOM_KIT"
  echo "           - Custom PhysX link: $USE_CUSTOM_PHYSX"
  echo "           - Build: $(normalize_bool "$SIM_BUILD")"
else
  echo "  • Sim:   skipped"
fi

if in_steps lab; then
  plan_line "Lab" "$LAB_SSH" "$LAB_DIR" "$LAB_BRANCH" "$LAB_COMMIT"
  echo "           - Create symlink: lab/_isaac_sim -> sim/_build/${platform}/${BUILD_CONFIG}"
  echo "           - Ensure Conda present (offer Miniconda install if missing)"
  echo "           - ./isaaclab.sh -c $LAB_ENV_NAME && ./isaaclab.sh -i"
  if truthy "$LAB_RECREATE"; then
    echo "           - Recreate env: $LAB_ENV_NAME"
  fi
  if truthy "$LAB_USE_PHYSX_EXTS"; then
    echo "           - Patch Lab app_launcher.py to include PhysX exts devFolders"
  fi
else
  echo "  • Lab:   skipped"
fi

echo

# Confirmation
confirm_and_go() {
  if truthy "$DRY_RUN"; then
    warn "Dry run: no commands will be executed."
    exit 0
  fi
  if truthy "$ASSUME_NO"; then
    warn "Cancelled by --no"
    exit 1
  fi
  if truthy "$ASSUME_YES"; then
    ok "Proceeding (non-interactive)"
    return 0
  fi
  if [[ -t 0 ]]; then
    read -r -p "Proceed with these actions? [y/N]: " answer
    case "${answer,,}" in
      y|yes) ok "Proceeding";;
      *) warn "Cancelled"; exit 1;;
    esac
  else
    die "No TTY to confirm. Re-run with -y to proceed non-interactively."
  fi
}
confirm_and_go

# PHYSX
if in_steps physx; then
  clone_or_prepare "$PHYSX_SSH" "$PHYSX_DIR"
  checkout_ref "$PHYSX_DIR" "$PHYSX_BRANCH" "$PHYSX_COMMIT" "PhysX"
  truthy "$PHYSX_BUILD" && build_physx || warn "PhysX build=false — not building"
else
  warn "PhysX step skipped"
fi

# KIT
if in_steps kit; then
  clone_or_prepare "$KIT_SSH" "$KIT_DIR"
  checkout_ref "$KIT_DIR" "$KIT_BRANCH" "$KIT_COMMIT" "Kit"
  truthy "$KIT_BUILD" && build_kit || warn "Kit build=false — not building"
else
  warn "Kit step skipped"
fi

# SIM
if in_steps sim; then
  clone_or_prepare "$SIM_SSH" "$SIM_DIR"
  checkout_ref "$SIM_DIR" "$SIM_BRANCH" "$SIM_COMMIT" "Isaac Sim"
  build_sim
else
  warn "Sim step skipped"
fi

# LAB
if in_steps lab; then
  clone_or_prepare "$LAB_SSH" "$LAB_DIR"
  checkout_ref "$LAB_DIR" "$LAB_BRANCH" "$LAB_COMMIT" "Isaac Lab"
  build_lab
else
  warn "Lab step skipped"
fi

# Final report
echo
ok "Build finished."
info "Checked out refs:"
[[ -d "$PHYSX_DIR/.git" ]] && (cd "$PHYSX_DIR" && echo "  PhysX    $(git rev-parse --short HEAD)") || echo "  PhysX    (skipped or missing)"
[[ -d "$KIT_DIR/.git"   ]] && (cd "$KIT_DIR"   && echo "  Kit      $(git rev-parse --short HEAD)") || echo "  Kit      (skipped or missing)"
[[ -d "$SIM_DIR/.git"   ]] && (cd "$SIM_DIR"   && echo "  Sim      $(git rev-parse --short HEAD)") || echo "  Sim      (skipped or missing)"
[[ -d "$LAB_DIR/.git"   ]] && (cd "$LAB_DIR"   && echo "  Lab      $(git rev-parse --short HEAD)") || echo "  Lab      (skipped or missing)"
echo
info "Locations:"
[[ -d "$PHYSX_DIR" ]] && echo "  $PHYSX_DIR" || echo "  $PHYSX_DIR (skipped)"
[[ -d "$KIT_DIR"   ]] && echo "  $KIT_DIR"   || echo "  $KIT_DIR (skipped)"
[[ -d "$SIM_DIR"   ]] && echo "  $SIM_DIR"   || echo "  $SIM_DIR (skipped)"
[[ -d "$LAB_DIR"   ]] && echo "  $LAB_DIR"   || echo "  $LAB_DIR (skipped)"
echo
ok "All done."
