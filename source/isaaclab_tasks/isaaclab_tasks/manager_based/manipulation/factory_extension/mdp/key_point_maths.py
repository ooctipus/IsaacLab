import torch
from typing import Tuple, Literal
from isaaclab.utils import math as math_utils
from ..state_machine.utils import compute_pose_error


@torch.jit.script
def cartesian_pairwise(
    set1: torch.Tensor, set2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the batched Cartesian product of two sets, with optional masking.

    For each batch index :math:`b`, let
    :math:`S^1_b = \{\,x_{b,i}\mid i=1,\dots,N1\}` with validity mask :math:`m^1_{b,i}`
    and
    :math:`S^2_b = \{\,y_{b,j}\mid j=1,\dots,N2\}` with validity mask :math:`m^2_{b,j}`.

    This function returns three tensors of shape :math:`[B, N1\times N2, *]`:
    - **out1** where each row :math:`(i,j)` is a copy of :math:`x_{b,i}`,
    - **out2** where each row :math:`(i,j)` is a copy of :math:`y_{b,j}`,
    - **pair_mask** which is True exactly when both :math:`m^1_{b,i}` and :math:`m^2_{b,j}` are True.

    Mathematically:
        :math:`out1[b,\,i\cdot N2 + j, :] = x_{b,i,:}`,
        :math:`out2[b,\,i\cdot N2 + j, :] = y_{b,j,:}`,
        :math:`pair\_mask[b,\,i\cdot N2 + j] = m^1_{b,i} \land m^2_{b,j}`.

    Args:
        set1:   Tensor of shape `(B, N1, D)`, the first batched set of vectors.
        set2:   Tensor of shape `(B, N2, D)`, the second batched set of vectors.
        mask1:  BoolTensor of shape `(B, N1)`, True where entries in `set1` are valid.
        mask2:  BoolTensor of shape `(B, N2)`, True where entries in `set2` are valid.

    Returns:
        Tuple of three tensors:
        - out1:      Tensor of shape `(B, N1*N2, D)`, repeated rows from `set1`.
        - out2:      Tensor of shape `(B, N1*N2, D)`, repeated rows from `set2`.
        - pair_mask: BoolTensor of shape `(B, N1*N2)`, True for valid `(i,j)` pairs.
    """
    B, N1, D = set1.size()
    _, N2, _ = set2.size()

    # 1) broadcast to [B, N1, N2, D]
    s1_bc = set1.unsqueeze(2).expand(B, N1, N2, D)
    s2_bc = set2.unsqueeze(1).expand(B, N1, N2, D)

    # 2) pairwise mask [B, N1, N2]
    pm = (mask1.unsqueeze(2) & mask2.unsqueeze(1))

    # 3) reshape into [B, N1*N2, *]
    out1 = s1_bc.reshape(B, N1 * N2, D)
    out2 = s2_bc.reshape(B, N1 * N2, D)
    outm = pm.reshape(B, N1 * N2)

    return out1, out2, outm


@torch.jit.script
def minimum_error_pair_selection(
    setA: torch.Tensor,
    maskA: torch.Tensor,
    setB: torch.Tensor,
    maskB: torch.Tensor,
    pose_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    For each batch index b, finds the pair (i, j) ∈ {0…NA-1}×{0…NB-1} that minimizes the
    weighted L2 pose-error between setA[b,i] and setB[b,j].

    Concretely, define for each candidate pair
        e_{b,i,j} = ∥ W · error( A_{b,i}, B_{b,j} ) ∥₂,
    where
        - A_{b,i}, B_{b,j} ∈ ℝ⁷ are (position, quaternion) poses,
        - error(·,·) ∈ ℝ⁶ produces (Δx,Δy,Δz,Δq_w,Δq_x,Δq_y,Δq_z) split into 3+3,
        - W = diag(pose_mask[b]) ∈ ℝ⁶ applies per-batch weighting,
        - ∥·∥₂ is the Euclidean norm.

    Then pick
        (i*, j*) = argmin_{i,j} e_{b,i,j}.

    Args:
        setA:      Tensor of shape (B, NA, 7), pose candidates in set A.
        maskA:     BoolTensor of shape (B, NA), True where A entries are valid.
        setB:      Tensor of shape (B, NB, 7), pose candidates in set B.
        maskB:     BoolTensor of shape (B, NB), True where B entries are valid.
        pose_mask: Tensor of shape (B, 6), per-batch weights for the 6 error components.

    Returns:
        bestA:   Tensor of shape (B, 7), the selected pose from setA (A_{b,i*}).
        idxA:    LongTensor of shape (B,), the chosen index i* into setA.
        bestB:   Tensor of shape (B, 7), the selected pose from setB (B_{b,j*}).
        idxB:    LongTensor of shape (B,), the chosen index j* into setB.
    """
    # 1) form all pairwise candidates
    sA, sB, pair_mask = cartesian_pairwise(setA, setB, maskA, maskB)
    B, P, _ = sA.size()    # P = NA * NB

    # 2) compute pose-errors only on valid pairs
    flatA = sA.view(B * P, 7)
    flatB = sB.view(B * P, 7)
    valid = pair_mask.view(B * P)
    pose_masking = pose_mask.view(B, 1, 6)

    pos_d, rot_d = compute_pose_error(flatA[valid, :3], flatA[valid, 3:], flatB[valid, :3], flatB[valid, 3:])
    raw_err = torch.cat([pos_d, rot_d], dim=1)      # [M_valid, 6]

    all_err = torch.zeros(B * P, 6, device=setA.device)
    all_err[valid] = raw_err
    errs = all_err.view(B, P, 6)

    # 3) apply W and compute L2 norm
    pos_e = errs[..., :3].abs() * pose_masking[..., :3]
    rot_e = errs[..., 3:].abs() * pose_masking[..., 3:]
    total = pos_e.norm(2, -1) + rot_e.norm(2, -1)
    total = total.masked_fill(~pair_mask, float('inf'))

    # 4) find argmin for each batch
    best_pair = torch.argmin(total, dim=1)  # [B]

    # 5) gather the best poses
    bestA = sA[torch.arange(B), best_pair]  # [B,7]
    bestB = sB[torch.arange(B), best_pair]  # [B,7]

    # 6) split flat index into i* and j*
    NB = setB.size(1)
    idxA = best_pair // NB                 # i* in [0..NA−1]
    idxB = best_pair % NB                  # j* in [0..NB−1]

    return bestA, idxA, bestB, idxB


@torch.jit.script
def select_first_valid_idx(
    mask: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    # False→0, True→1 then argmax gives first True or 0 if none
    return mask.float().argmax(dim=dim)


@torch.jit.script
def select_random_valid_idx(
    mask: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    scores = torch.rand_like(mask, dtype=torch.float)
    scores = scores.masked_fill(~mask, -1.0)
    return scores.argmax(dim=dim)


@torch.jit.script
def _equal_shape_select_valid(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    strategy: str = "first"
) -> Tuple[torch.Tensor, torch.Tensor]:
    if strategy == "first":
        idx = select_first_valid_idx(mask, dim)
    else:
        idx = select_random_valid_idx(mask, dim)

    idx_exp = idx.unsqueeze(dim)
    picked_vals = values.gather(dim, idx_exp).squeeze(dim)
    picked_mask = mask.gather(dim, idx_exp).squeeze(dim)
    return picked_vals, picked_mask


@torch.jit.script
def _equal_prefix_select_valid(
    flat_vals: torch.Tensor,   # shape (P, N, S)
    flat_mask: torch.Tensor,   # shape (P, N)
    strategy: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pick along axis=1
    if strategy == "random":
        idx = select_random_valid_idx(flat_mask, dim=1)
    else:
        idx = select_first_valid_idx(flat_mask, dim=1)

    rows = torch.arange(flat_mask.size(0), device=flat_mask.device)
    picked_mask = flat_mask[rows, idx]    # → (P,)
    picked_vals = flat_vals[rows, idx]    # → (P, S)
    return picked_vals, picked_mask


def select_valid(
    values: torch.Tensor,   # shape: P₁×…×Pᵢ × N × S₁×…×Sⱼ
    mask: torch.Tensor,   # shape: P₁×…×Pᵢ × N
    dim: int = -1,  # which axis is “N”
    strategy: Literal["first", "random"] = "first"  # "first" or "random"
) -> Tuple[torch.Tensor, torch.Tensor]:
    # normalize dim
    if values.dim() == mask.dim():
        return _equal_shape_select_valid(values, mask, dim, strategy)
    else:
        d = dim if dim >= 0 else mask.dim() + dim

        # flatten prefixes [0..d-1] → a single leading axis
        flat_mask = mask.flatten(0, d - 1)    # → (P, N)
        flat_vals = values.flatten(0, d - 1)  # → (P, N, …)

        # flatten all suffix dims [d+1..] of flat_vals into one trailing dim
        if flat_vals.dim() > 2:
            # merge dims 2..end → suffix size S
            flat_vals = flat_vals.flatten(2, -1)  # → (P, N, S)

        # call the JIT picker over these flattened arrays
        picked_flat, picked_mask = _equal_prefix_select_valid(flat_vals, flat_mask, strategy)  # shapes (P, S), (P,)

        # now reshape back to (P₁…Pᵢ, S₁…Sⱼ) and (P₁…Pᵢ)
        prefix_shape = tuple(mask.size()[:d])           # (P₁,…,Pᵢ)
        suffix_shape = tuple(values.size()[d + 1:])       # (S₁,…,Sⱼ)

        out_vals = picked_flat.reshape(*prefix_shape, *suffix_shape)
        out_mask = picked_mask.reshape(*prefix_shape)
        return out_vals, out_mask


@torch.jit.script
def z_axis_alignment_filter(
    values: torch.Tensor,     # Shape: (n_envs, ..., 7)
    item_mask: torch.Tensor,  # Shape: (n_envs, ...) - Item-level activity mask
    z_axis: torch.Tensor,     # Shape: (n_envs, 3) - Per-environment reference z-axis
    threshold: float
) -> torch.Tensor:              # Output Shape: (n_envs, ...)

    updated_item_level_mask = item_mask.clone()  # Will be modified based on filter results

    # Call the JIT-compatible quaternion to matrix function.
    # Replace `matrix_from_quat_for_jit_example` with your JIT-scripted `math_utils.matrix_from_quat`
    active_quats = values[..., 3:7][item_mask]
    z_axis1_active = math_utils.matrix_from_quat(active_quats)[..., 2] # Assuming 3rd row is z-axis
    norm_val = torch.linalg.norm(z_axis1_active, dim=-1, keepdim=True)
    local_axis_active = z_axis1_active / norm_val.clamp(min=1e-6)

    # Get the corresponding global z_axis for each active item
    env_indices = torch.arange(values.shape[0], device=values.device)

    # Construct view_shape for env_indices. JIT handles list ops for shapes.
    view_shape_list = [-1] + [1] * (item_mask.ndim - 1)

    # If item_mask is all False, active_env_ids will be an empty tensor.
    active_env_ids = env_indices.view(view_shape_list).expand_as(item_mask)[item_mask]
    z_axis_for_active = z_axis[active_env_ids]  # Results in empty tensor if active_env_ids is empty

    # Perform dot product and apply threshold
    # If inputs are empty, dots and passed_filter_for_active will also be empty.
    dots = (local_axis_active * z_axis_for_active).sum(dim=-1)
    passed_filter_for_active = dots >= threshold

    if item_mask.any():  # This condition ensures we only attempt assignment if there are places to assign.
        updated_item_level_mask[item_mask] = passed_filter_for_active

    return updated_item_level_mask
