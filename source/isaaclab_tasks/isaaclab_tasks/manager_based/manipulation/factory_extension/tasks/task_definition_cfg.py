from dataclasses import MISSING
from typing import Sequence
from isaaclab.utils import configclass

from .assembly_key_points_cfg import KeyPointCfg


@configclass
class KeyPoints:

    def __post_init__(self):
        for key, val in self.__dict__.items():
            if isinstance(val, list):
                setattr(self, key, {str(i): v for i, v in enumerate(val)})

    def ordered_points(self) -> Sequence[KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg]]:
        """return the list of all KeyPointCfg instances, in order."""
        return list(self.__dict__.values())

    def ordered_names(self) -> list[str]:
        """return the list of all KeyPointCfg instances' name, in order."""
        return list(self.__dict__.keys())


@configclass
class BaseTaskKeyPointCfg:
    key_points: KeyPoints = MISSING  # type:ignore

    def asset_set(self) -> set[str]:
        """Collect all `root.object_collection_names` from each key‐point."""
        names = []
        for kp in self.key_points.ordered_points():
            coll = kp.root.object_collection_names
            assert coll is not None, "each KeyPointCfg must define .root.object_collection_names"
            if isinstance(coll, str):
                coll = [coll]
            names.extend(coll)
        return set(names)


@configclass
class SuccessCondition:
    pos_components: str = "xyz"
    rot_components: str = "rpy"
    pos_std: tuple[float, float, float] = (0.003, 0.003, 0.003)  # 3 mm
    rot_std: tuple[float, float, float] = (0.025, 0.025, 0.025)  # 1.43 degree
    pos_threshold: tuple[float, float, float] = (0.003, 0.003, 0.003)  # 3 mm
    rot_threshold: tuple[float, float, float] = (0.025, 0.025, 0.025)  # 1.43 degree

    def pose_components_mask(self):
        components = [0, 0, 0, 0, 0, 0]
        i = 0
        for component in "xyz":
            if component in self.pos_components:
                components[i] = 1
                i += 1
        for component in "rpy":
            if component in self.rot_components:
                components[i] = 1
                i += 1

        return components


@configclass
class TaskKeyPointCfg(BaseTaskKeyPointCfg):
    @configclass
    class TaskKeyPoints(KeyPoints):
        asset_align: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        asset_align_against: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore

    key_points: TaskKeyPoints = MISSING  # type:ignore
    success_condition: SuccessCondition = SuccessCondition()


@configclass
class ManipulationKeyPointCfg(BaseTaskKeyPointCfg):
    @configclass
    class ManipulationKeyPoints(KeyPoints):
        robot_root: KeyPointCfg | list[KeyPointCfg] = MISSING  # type:ignore
        robot_object_held: KeyPointCfg | list[KeyPointCfg] = MISSING  # type:ignore
        asset_grasp: KeyPointCfg | list[KeyPointCfg] = MISSING  # type:ignore

    key_points: ManipulationKeyPoints = MISSING  # type:ignore
    held_asset_diameter: float = MISSING  # type:ignore
    success_condition: SuccessCondition = SuccessCondition()


@configclass
class AssetResetOnTaskBoardKeyPointCfg(BaseTaskKeyPointCfg):
    """KeyPointCfg for the asset reset pose."""

    @configclass
    class AssetResetKeyPoints(KeyPoints):
        bnc_plug: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        bnc_socket: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        dsub_plug: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        dsub_socket: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rj45_plug: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rj45_socket: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        waterproof_plug: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        waterproof_socket: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        usba_plug: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        usba_socket: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        nut_m4: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        bolt_m4: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        nut_m8: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        bolt_m8: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        nut_m12: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        bolt_m12: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        nut_m16: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        bolt_m16: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_peg_4mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_hole_4mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_peg_8mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_hole_8mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_peg_12mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_hole_12mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_peg_16mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rectangular_hole_16mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rod_4mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        hole_4mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rod_8mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        hole_8mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rod_12mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        hole_12mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        rod_16mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        hole_16mm: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        large_gear: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        medium_gear: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        small_gear: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore
        gear_base: KeyPointCfg | list[KeyPointCfg] | dict[str, KeyPointCfg] = MISSING  # type:ignore

    key_points: AssetResetKeyPoints = MISSING  # type:ignore
