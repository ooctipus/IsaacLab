from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


@configclass
class Offset:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def pose(self):
        return self.pos + self.quat

    def apply(self, root: RigidObject | Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        data = root.data.root_pos_w
        pos_w, quat_w = math_utils.combine_frame_transforms(
            root.data.root_pos_w,
            root.data.root_quat_w,
            torch.tensor(self.pos).to(data.device).repeat(data.shape[0], 1),
            torch.tensor(self.quat).to(data.device).repeat(data.shape[0], 1),
        )
        return pos_w, quat_w

    def combine(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w,
            quat_w,
            torch.tensor(self.pos).to(pos_w.device).repeat(pos_w.shape[0], 1),
            torch.tensor(self.quat).to(pos_w.device).repeat(pos_w.shape[0], 1),
        )
        return pos_w, quat_w

    def subtract(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offset_pos = torch.tensor(self.pos).to(pos_w.device).repeat(pos_w.shape[0], 1)
        offset_quat = torch.tensor(self.quat).to(pos_w.device).repeat(pos_w.shape[0], 1)
        inv_offset_pos = -math_utils.quat_apply(math_utils.quat_inv(offset_quat), offset_pos)
        inv_offset_quat = math_utils.quat_inv(offset_quat)
        return math_utils.combine_frame_transforms(pos_w, quat_w, inv_offset_pos, inv_offset_quat)


@configclass
class KeyPointsTableTopHole:
    hole0_tip_offset: Offset = Offset(pos=(0.05625, 0.05625, 0.015565))
    hole0_leg_assembled_offset: Offset = Offset(pos=(0.05625, 0.05625, -0.009435))


@configclass
class KeyPointsTableLeg:
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, -0.056658))
    graspable: Offset = Offset(pos=(0.0, 0.0, 0.015), quat=(1.0, 0.0, 0.0, 0.0))
    diameter: float = 0.03


@configclass
class KeyPointsRobotiqGripper:
    offset: Offset = Offset(pos=(0.1345, 0.0, 0.0), quat=(0.70711, 0.0, -0.70711, 0.0))


KEYPOINTS_TABLETOPHOLE = KeyPointsTableTopHole()
KEYPOINTS_TABLELEG = KeyPointsTableLeg()
KEYPOINTS_ROBOTIQGRIPPER = KeyPointsRobotiqGripper()
