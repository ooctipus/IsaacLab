# UR5 Furniture Assembly Training and Distillation Guide

This guide provides a comprehensive walkthrough for training a UR5 robot to perform furniture assembly tasks using the isaaclab framework, followed by distillation to a vision-based diffusion policy.

## Table of Contents

1. [Collect Initial States](#1-collect-initial-states)
2. [Visualize Initial States](#2-visualize-initial-states)
3. [Train with Initial States](#3-train-with-initial-states)
4. [Training Commands](#4-training-commands)
5. [Evaluate Policy](#5-evaluate-policy)
6. [Collect RGB Data](#6-collect-rgb-data)
7. [Train ResNet-MLP](#7-train-resnet-mlp)
8. [Evaluate ResNet-MLP](#8-evaluate-resnet-mlp)

---

## 1. Collect Initial States

### Insertion Task Only

Collect initial states for the insertion phase:

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Insertion-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/insertion_init_states_dataset.hdf5
```

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-AssembledGrasped-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/assembledgrasped_init_states_dataset.hdf5
```

### Complete Task (All Components)

For the full furniture assembly pipeline, collect data for all phases:

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Reaching-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/reaching_init_states_dataset.hdf5
```

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Grasped-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/grasped_init_states_dataset.hdf5
```

---

## 2. Visualize Initial States

### Insertion Task

```bash
python scripts_v2/tools/visualize_saved_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-v0 \
    --dataset_file ./datasets/insertion_init_states_dataset_preprocessed.pt \
    --num_envs 4
```

```bash
python scripts_v2/tools/visualize_saved_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-v0 \
    --dataset_file ./datasets/assembledgrasped_init_states_dataset_preprocessed.pt \
    --num_envs 4
```

### Complete Task

```bash
python scripts_v2/tools/visualize_saved_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-v0 \
    --dataset_file ./datasets/reaching_init_states_dataset_preprocessed.pt \
    --num_envs 4
```

```bash
python scripts_v2/tools/visualize_saved_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-v0 \
    --dataset_file ./datasets/grasped_init_states_dataset_preprocessed.pt \
    --num_envs 4
```

---

## 3. Train with Initial States

Modify the configuration file at:
`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/furniture_bench/config/ur5/one_leg_ur5.py`

### Insertion Task Configuration

```python
combined_reset = EventTerm(
    func=task_mdp.MultiResetManager,
    mode="reset",
    params={
        "datasets": [
            "furniture_datasets/insertion_init_states_preprocessed.pt",
            "furniture_datasets/assembled_grasped_init_states_preprocessed.pt"
        ],
        "probs": [0.5, 0.5],
        "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        "failure_rate_sampling": False,
        "bin_samples": True
    }
)
```

### Complete Task Configuration

Use equal sampling (25% each) across all four datasets:

```python
combined_reset = EventTerm(
    func=task_mdp.MultiResetManager,
    mode="reset",
    params={
        "datasets": [
            # "furniture_datasets/reaching_init_states_preprocessed.pt",
            # "furniture_datasets/grasped_init_states_preprocessed.pt",
            "furniture_datasets/insertion_init_states_preprocessed.pt",
            "furniture_datasets/assembled_grasped_init_states_preprocessed.pt"
        ],
        "probs": [0.25, 0.25],
        "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        "failure_rate_sampling": False,
        "bin_samples": True
    }
)
```

---

## 4. Training Commands

### Single GPU Training

```bash
python scripts_v2/rl/main.py \
    --task Pat-OneLeg-Ur5-JointPosRel-v0 \
    --num_envs 16384 \
    --job_type train \
    --rl_framework rslrl \
    --logger wandb \
    --headless
```

### Distributed GPU Training

**Recommended for complete task training** (requires at least 48k environments):

```bash
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts_v2/rl/main.py \
    --task Pat-OneLeg-Ur5-JointPosRel-v0 \
    --num_envs 16384 \
    --job_type train \
    --rl_framework rslrl \
    --logger wandb \
    --headless \
    --distributed
```

### Finetune Policy (clipped actions, no dense reward)
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts_v2/rl/main.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Finetune-v0 \
    --num_envs 8192 \
    --job_type train \
    --rl_framework rslrl \
    --logger wandb \
    --headless \
    --distributed \
    --checkpoint_dir=logs/rsl_rl/Pat-OneLeg-Ur5-JointPosRel-v0/2025-08-27_21-09-49/model_2900.pt
```

---

## 5. Evaluate Policy

Export the trained policy to JIT/ONNX format:

```bash
python scripts_v2/rl/main.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Finetune-Eval-v0 \
    --num_envs 4 \
    --job_type eval \
    --rl_framework rslrl \
    --checkpoint_dir logs/model_2600.pt
```

---

## 6. Collect RGB Data

### Record RGB Init States

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Reaching-RGB-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/reaching_init_states_rgb_dataset.hdf5
```

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Grasped-RGB-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/grasped_init_states_rgb_dataset.hdf5
```

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Insertion-RGB-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/insertion_init_states_rgb_dataset.hdf5
```

```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-AssembledGrasped-RGB-v0 \
    --num_envs 15000 \
    --headless \
    --num_initial_conditions 10000 \
    --dataset_file furniture_datasets/assembled_grasped_init_states_rgb_dataset.hdf5
```

### Data Collection

```bash
python scripts_v2/tools/collect_demos_rsl_rl_policy.py \
    --task Pat-OneLeg-Ur5-JointPosRel-DataCollection-RGB-v0 \
    --num_envs 96 \
    --headless \
    --dataset_file "./datasets/one_leg_rgb.zarr" \
    --num_demos 10000 \
    --enable_cameras \
    agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["logs/exported/policy.pt"]
```

**Performance Note:** 96 environments fits on a 48GB GPU machine and takes ~1 day to collect 10K trajectories on an A40 GPU.

---

## 7. Train ResNet-MLP

### Setup

**One-time setup required:**

1. Clone the diffusion policy repository:
   ```bash
   git clone git@github.com:patrickhaoy/diffusion_policy.git
   cd diffusion_policy
   git checkout pat/dev
   ```

2. Ensure the folder hierarchy looks like:
   ```
   isaaclab/
   diffusion_policy/
   ```

3. Create and activate the conda environment (follow README.md instructions and install conda_environment.yaml)

4. Install additional dependencies:
   ```bash
   pip install huggingface_hub==0.16.4
   ```

### Training Command

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    train.py \
    --config-name train_mlp_sim2real_image_workspace.yaml \
    --config-dir diffusion_policy/config \
    task.dataset.dataset_path=/path/to/your/isaaclab/datasets/one_leg_rgb.zarr
```

### Configuration Notes

- **GPU Processes**: Adjust `--num_processes=2` based on available GPUs
- **Batch Size**: Set according to GPU memory (batch size 128 barely fits in 24GB GPU)
- **CPU Workers**: Ensure `num_workers` × `num_processes` ≤ available CPU cores

### Dataset Configuration

In `diffusion_policy/config/task/sim2real_image.yaml`, configure `dataset.use_cache`:

- **`False`**: Load data from disk (slower, less memory usage)
- **`True`**: Load data into CPU memory (faster, higher memory usage)

---

## 8. Evaluate ResNet-MLP

### Setup

**One-time setup required:**

1. Navigate to diffusion policy repository
2. Install dependencies:
   ```bash
   pip install -e .
   pip install imagecodecs av zarr accelerate huggingface_hub==0.16.4
   ```

### Collect initial states for eval
```bash
python scripts_v2/tools/record_init_states.py \
    --task Pat-OneLeg-Ur5-JointPosRel-Reaching-v0 \
    --num_envs 1000 \
    --headless \
    --num_initial_conditions 1000 \
    --dataset_file datasets/init_stable_states_dataset.hdf5
```

### Evaluation Command

```bash
python scripts_v2/tools/eval_diffusion_policy.py \
    --task Pat-OneLeg-Ur5-JointPosAbs-Eval-RGB-v0 \
    --num_envs 16 \
    --enable_cameras \
    --headless \
    --checkpoint /path/to/your/checkpoint.ckpt \
    --save_video
```

**Performance Note:** Evaluation with 16 environments fits on a 24GB GPU. 