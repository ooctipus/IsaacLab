# Cluster setup on a new machine

Use this branch as an entry point for cluster instructions. Cluster tooling lives in the private
[ooctipus/isaaclab-cluster](https://github.com/ooctipus/isaaclab-cluster) repository and works with any training
branch. You do not need to rebase or train from this `cluster` branch. The older `cluster/`, `submit.sh`, and
`buildnpush.sh` implementations retained on this branch are superseded for this workflow.

## 1. Prepare the submitting machine

These steps assume a Linux host with Bash, Git, OpenSSH, curl, and Python 3.11 or newer. The host does not need
a GPU, Docker, or the training environment. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and the [GitHub CLI](https://cli.github.com/), then sign into the GitHub account that can read both private
cluster tooling and your training repositories:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

An agent can run the setup and pause at a browser/device login for the human to complete. Each new host needs
its own GitHub and cluster-service login; tokens are not committed or copied from this branch.

For RTX6K, follow the internal
[SkyPilot runbook](https://docs.google.com/document/d/15f0BkE3YBvSxETqQA1izcV2Oh5-Bsi2GBLVghAAigEI/edit).
Its enterprise client installer is:

```bash
curl -fsSL https://skypilot-cli.s3.us-west-2.amazonaws.com/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
sky --version
unset SKYPILOT_API_SERVER_ENDPOINT
sky api login -e https://skypilot.gear.nvidia.com
sky api info
sky workspace info
```

Use the enterprise client, including support for `sky jobs launch --secret`, and confirm the shared API is
healthy with your expected account. Access to the `gear-skypilot-access` group/enterprise application and
membership in `akamai-rtx6k-isaac-lab` are separate requirements. Ask the service/project administrator if either
is missing. For a terminal without a local browser, add `--no-browser` to the login command. If TLS verification
fails, follow the runbook's NVIDIA CA bundle instructions and persist `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE`.
The [allocation document](https://docs.google.com/document/d/183l6dOqoPLWt_0nlsGVjZlojXGBASS1DH6aErs_ncf4/edit)
explains team workspace access. The launcher passes the Isaac Lab workspace and Akamai context explicitly.

For OSMO, install its CLI and complete its normal browser login:

```bash
curl -fsSL https://us-west-2-aws.osmo.nvidia.com/cli/install.sh | bash
osmo login https://us-west-2-aws.osmo.nvidia.com
osmo pool list
```

Create an SSH key if the new host does not already have one. OSMO allocations default to `~/.ssh/id_ed25519`;
use `--identity /ABS/PATH/KEY` when choosing a different key. Keep an existing key instead of overwriting it.

```bash
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519"
fi
```

If the key has a passphrase, load it into your SSH agent with `ssh-add ~/.ssh/id_ed25519` before allocating
so the launcher can connect without an interactive key prompt.

## 2. Install the independent workspace launcher

The following is a fresh-machine layout; adapt the parent directory if needed. If a checkout, launcher, or
workspace instructions already exist, preserve them and inspect/update them instead of repeating these writes.

```bash
mkdir -p "$HOME/Projects/IsaacLab.wt"
gh repo clone ooctipus/isaaclab-cluster "$HOME/Projects/isaaclab-cluster"
ln -s ../isaaclab-cluster/bin/cluster "$HOME/Projects/IsaacLab.wt/cluster"
cd "$HOME/Projects/IsaacLab.wt"
./cluster --help
```

`./cluster` delegates to the separate repository and uv installs only the small launcher's dependencies on the
host. It does not install Isaac Lab locally. To update tooling later, use
`git -C "$HOME/Projects/isaaclab-cluster" pull --ff-only`.

On a fresh workspace, create `IsaacLab.wt/AGENTS.md` with this routing instruction. On an existing workspace,
merge it into the existing file instead of replacing other instructions:

```markdown
# Workspace cluster workflow

Use ./cluster here for GPU batch jobs and development allocations.
Read ../isaaclab-cluster/README.md for setup and commands, and its AGENTS.md before editing tooling.
Keep orchestration in that separate repository; do not copy it into training worktrees or build images.
Use --repo URL --ref BRANCH to select pushed source; GitHub login, clone, and uv sync run automatically.
Require --duration or --until for development allocations. Reconnect with ./cluster ssh NAME.
Never put tokens in arguments/manifests or cancel existing sessions without authorization.
Follow each training worktree's own AGENTS.md for changes to its source.
```

A training checkout is optional on the submitting host. To obtain this instruction branch separately:

```bash
gh repo clone ooctipus/IsaacLab "$HOME/Projects/IsaacLab.wt/cluster-guide" -- --branch cluster --single-branch
```

## 3. Allocate or submit

Run from `IsaacLab.wt`. Replace `YOUR_TRAINING_BRANCH` with a pushed branch of `ooctipus/IsaacLab`, or pass
`--repo URL` for another repository. Push local edits before launching; source comes from GitHub.

```bash
./cluster allocate --target rtx6k --gpus 8 --duration 8h --ref YOUR_TRAINING_BRANCH
./cluster allocate --target isaac-lab-l40s-03 --gpus 8 --duration 8h --ref YOUR_TRAINING_BRANCH
./cluster ssh NAME
./cluster status NAME
./cluster cancel NAME
```

RTX6K requires eight GPUs per node. OSMO accepts an accessible pool name and discovers its platform/resources.
The printed allocation name is what replaces `NAME`. To keep an allocation through a month's final day, use
midnight at the start of the following month, e.g. `--until '2026-12-01T00:00:00'` through November 30 in the
default `America/Los_Angeles` timezone. Deadlines include queue/setup time; service teardown can have a grace period.

The default uses the service/public base, installs GitHub CLI, uv, SSH, Nsight Systems, W&B, Claude Code, and
other dev tools, reuses the host's GitHub login, clones, and runs `uv sync --locked`. No custom image build or
private-image pull secret is needed. Startup takes longer because each fresh allocation installs its tools.
Select branch-specific extras with `--sync-args '--locked --extra all --extra importers --extra wandb'` only when
that branch supports them. Claude Console login is explicit: `./cluster auth NAME claude --console`.
W&B credentials are reused from the host when available; otherwise use `./cluster auth NAME wandb`.

For unattended work, use the same source options with `submit` and put the command after `--`:

```bash
./cluster submit --target rtx6k --gpus 8 --duration 8h --ref YOUR_TRAINING_BRANCH -- \
  uv run --no-sync python path/to/train.py
```

The launcher automatically supplies GitHub access through the backend's secret mechanism for batch jobs.
Read the maintained [tooling README](https://github.com/ooctipus/isaaclab-cluster#readme) for credential storage,
project extras, distributed submission, source defaults, cache persistence, and troubleshooting. That repository
owns operational behavior; this branch only explains how to find and install it on another machine.
