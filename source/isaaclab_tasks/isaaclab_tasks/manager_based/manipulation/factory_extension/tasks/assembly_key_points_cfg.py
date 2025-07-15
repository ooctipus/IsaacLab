from dataclasses import MISSING
import torch
from typing import Iterable, Union
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
import re


def to_snake(s: str) -> str:
    # 1) split before Upper→lower transitions, but not inside all-caps runs
    s = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', s)
    # 2) split at lower→Upper transitions
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', s)
    # 3) split off a digit only if it's after a lowercase letter
    s = re.sub(r'(?<=[a-z])(?=\d)', '_', s)
    return s.lower()


def to_pascal(s: str, acronyms: dict[str, str] | None = None) -> str:
    if acronyms is None:
        acronyms = {
            "mm": "MM",
            "usba": "USBA",
            "bnc": "BNC",
            "dsub": "DSUB",
            "rj": "RJ",
        }

    def _repl(m: re.Match) -> str:
        token = m.group(1)
        parts = re.compile(r'[A-Za-z]+|\d+').findall(token)
        out = []
        for part in parts:
            if part.isdigit():
                # digits stay as-is
                out.append(part)
            else:
                # letters: check for acronym or capitalize
                key = part.lower()
                out.append(acronyms.get(key, part.capitalize()))
        return "".join(out)

    # replace each "_chunk" (or leading chunk) with its Pascalized form
    return re.compile(r'(?:^|_)([A-Za-z0-9]+)').sub(lambda m: _repl(m), s)

@configclass
class Offset:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def pose(self) -> tuple[float, float, float, float, float, float, float]:
        return self.pos + self.quat


@configclass
class SymmetryOffsets:
    origin: Offset = Offset()
    """The origin of the symmetry as offset to root."""

    angles: list[list[float]] = [MISSING]  # type: ignore
    """The angles of the symmetry in radians, empty means full symmetry."""

    axis: list[tuple[float, float, float]] = MISSING  # type: ignore
    """The axis of the symmetry as a unit vector."""

    offset: Offset = Offset()


@configclass
class OffsetRange:
    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    z: tuple[float, float] = (0.0, 0.0)
    roll: tuple[float, float] = (0.0, 0.0)
    pitch: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (0.0, 0.0)


def _dedupe_rows(rows: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Remove duplicate pose/quaternion rows.

    Rows shape: (m,7) with [x,y,z, w,qx,qy,qz].
    """
    # 1) canonicalize quaternion sign so q and -q collapse (w >= 0)
    w = rows[:, 3:4]
    signs = torch.sign(w)
    signs[signs == 0] = 1.0
    rows[:, 3:] = rows[:, 3:] * signs
    # 2) quantize to remove tiny jitter
    rows = torch.round(rows / eps) * eps
    # 3) unique
    return torch.unique(rows, dim=0)


def symmetry_quat_group(
    axes: list[torch.Tensor],
    angles: list[torch.Tensor],
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Build the finite symmetry quaternion set, collapsing all perpendicular discrete-flips
    to a single representative when a continuous axis is present.
    """
    device = axes[0].device

    # 1) Detect continuous symmetry axes (those with zero angles)
    # has_continuous = any(len(ang) == 0 for ang in angles)

    # 2) Build list of strictly discrete (axis,angles) pairs
    discrete = [(ax, ang) for ax, ang in zip(axes, angles) if len(ang) > 0]

    # 3) If there’s a continuous axis, keep only the first discrete axis
    #    so you only generate identity + one flip
    # if has_continuous and discrete:
    #     discrete = [discrete[0]]

    # 4) If now there are no discrete axes, there’s only the identity
    if not discrete:
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 5) Otherwise, unzip and do the usual mesh‑grid
    axes_f, angles_f = zip(*discrete)
    grids = torch.meshgrid(*angles_f, indexing="ij")
    combos = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (Nprod, k)

    # 6) Compose each row of angles into a quaternion
    quats = []
    for row in combos:
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        for θ, ax in zip(row, axes_f):
            single_q = math_utils.quat_from_angle_axis(
                θ.unsqueeze(0),       # (1,)
                ax.unsqueeze(0)       # (1,3)
            )
            q = math_utils.quat_mul(single_q, q)
        quats.append(q)

    return torch.cat(quats, dim=0)  # (N prod,4)


def symmetry_group(
    axes: list[torch.Tensor],
    angles: list[torch.Tensor],
    symmetry_origin_pose: torch.Tensor,
    offset_pose: torch.Tensor
) -> torch.Tensor:
    """
    Given an origin frame and an object pose (both in root/world coordinates), generate
    all symmetric world space poses of the object under the specified finite rotations.

    Args:
        axes:                   List of k unit axes (each shape (3,)) defining symmetry directions.
        angles:                 List of k 1D tensors (Ni,) of rotation angles (radians) around each axis.
        symmetry_origin_pose:   Tensor (7,) = [pos(3), quat(4)] of the symmetry origin in root frame.
        offset_pose:            Tensor (7,) = [pos(3), quat(4)] of the object pose in root frame.

    Returns:
        world_poses: Tensor of shape (M, 7) containing the symmetric object poses in root frame,
                     with rows [pos(3), quat(4)].
    """
    # 1) Build symmetry quaternions
    sym_q = symmetry_quat_group(axes, angles)  # (M,4)
    M = sym_q.shape[0]

    # 2) Compute object pose relative to origin: local frame
    p_rel = offset_pose[:3] - symmetry_origin_pose[:3]  # (3,)
    q_o_inv = math_utils.quat_inv(symmetry_origin_pose[3:].unsqueeze(0)).squeeze(0)
    q_rel = math_utils.quat_mul(q_o_inv.unsqueeze(0), offset_pose[3:].unsqueeze(0)).squeeze(0)  # (4,)

    # 3) Apply each symmetry to the local pose
    p_rel_expand = p_rel.unsqueeze(0).expand(M, 3)
    q_rel_expand = q_rel.unsqueeze(0).expand(M, 4)
    rot_p = math_utils.quat_apply(sym_q, p_rel_expand)  # (M,3)
    rot_q = math_utils.quat_mul(sym_q, q_rel_expand)  # (M,4)

    # 4) Map symmetric local poses back to root frame
    origins = symmetry_origin_pose.unsqueeze(0).expand(M, 7)  # (M,7)
    root_pos, root_quat = math_utils.combine_frame_transforms(origins[:, :3], origins[:, 3:], rot_p, rot_q)  # (M,7)

    return torch.cat([root_pos, root_quat], dim=1)


@configclass
class KeyPointCfg:
    offsets: Iterable[Union[Offset, SymmetryOffsets]] = MISSING  # type:ignore
    root: SceneEntityCfg = MISSING  # type:ignore

    def offsets_group(self, device: str = "cpu") -> torch.Tensor:
        """
        Compute all equivalent offset poses (pos+quat) for this KeyPointCfg,
        including manual Offsets and SymmetryOffsets definitions.

        Returns:
            Tensor of shape (m, 7) where each row is [x, y, z, w, qx, qy, qz].
        """

        all_rows = []
        for entry in self.offsets:
            # Pure fixed offset
            if isinstance(entry, Offset):
                row = torch.tensor(entry.pose, device=device, dtype=torch.float32)
                all_rows.append(row.unsqueeze(0))

            # Symmetry‐generated offsets
            elif isinstance(entry, SymmetryOffsets):
                # No discrete angles = continuous or trivial: just one offset
                if not entry.angles:
                    row = torch.tensor(entry.offset.pose, device=device, dtype=torch.float32)
                    all_rows.append(row.unsqueeze(0))
                    continue

                # Build finite symmetry group about a single axis
                if isinstance(entry.axis[0], (float, int)):
                    axes_list = [torch.as_tensor(entry.axis, device=device, dtype=torch.float32)]
                    angles_list = [torch.as_tensor(entry.angles, device=device, dtype=torch.float32)]
                else:
                    axes_list = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in entry.axis]
                    angles_list = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in entry.angles]

                origin = torch.as_tensor(entry.origin.pose, device=device, dtype=torch.float32)
                offset = torch.as_tensor(entry.offset.pose, device=device, dtype=torch.float32)

                # Enumerate all poses under that symmetry
                poses = symmetry_group(axes_list, angles_list, origin, offset)  # (M,7)
                all_rows.append(poses)

        # Concatenate into a single (m,7) tensor
        if not all_rows:
            raise ValueError("No offsets defined in KeyPointCfg.")
        # Concatenate into a single (m,7) tensor
        rows = torch.cat(all_rows, dim=0)
        return _dedupe_rows(rows, eps=1e-6)


@configclass
class ObjectKeyPointsCfg:
    root: SceneEntityCfg | None = None

    def __post_init__(self):
        if self.root is None:
            object_name = to_snake(self.__class__.__name__.replace("KeyPointsCfg", ""))
            self.root = SceneEntityCfg("assets", object_collection_names=object_name)
        for value in self.__dict__.values():
            if isinstance(value, KeyPointCfg):
                value.root = self.root
