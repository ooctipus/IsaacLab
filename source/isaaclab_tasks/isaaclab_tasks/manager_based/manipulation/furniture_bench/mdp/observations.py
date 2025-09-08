import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.sensors import TiledCamera, Camera, RayCasterCamera

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset=None,
    root_asset_offset=None,
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)
    return torch.cat([target_pos_b, target_quat_b], dim=1)


def asset_link_velocity_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    taget_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    asset_lin_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_lin_vel_w[:, taget_body_idx].view(-1, 3),
    )
    asset_ang_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_lin_vel_w[:, taget_body_idx].view(-1, 3),
    )

    return torch.cat([asset_lin_vel_b, asset_ang_vel_b], dim=1)


def joint_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.joint_ids]
    return joint_forces.view(env.num_envs, -1)


def get_material_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_material_properties().view(env.num_envs, -1)


def get_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses().view(env.num_envs, -1)


def get_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff.view(env.num_envs, -1)


def get_joint_armature(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_armature.view(env.num_envs, -1)


def get_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness.view(env.num_envs, -1)


def get_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping.view(env.num_envs, -1)


def process_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    process_image: bool = True,
    output_size: tuple = (224, 224)
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        process_image: Whether to normalize the image. Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    assert data_type == "rgb", "Only RGB images are supported for now."
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type].clone()

    start_dims = torch.arange(len(images.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    current_size = (images.shape[s + 1], images.shape[s + 2])

    # Convert to float32 and normalize in-place
    images = images.to(dtype=torch.float32)  # Avoid redundant .float() and .type() calls
    images.div_(255.0).clamp_(0.0, 1.0)  # Normalize and clip in-place
    images = images.permute(start_dims + [s + 3, s + 1, s + 2])

    if current_size != output_size:
        # Perform resize operation
        images = F.interpolate(
            images,
            size=output_size,
            mode="bilinear",
            antialias=True
        )

    # rgb/depth image normalization
    if not process_image:
        # Reverse the permutation
        reverse_dims = torch.argsort(torch.tensor(start_dims + [s + 3, s + 1, s + 2]))
        images = images.permute(reverse_dims.tolist())
        # Convert back to uint8 in-place
        images.mul_(255.0).clamp_(0, 255)  # Scale and clamp in-place
        images = images.to(dtype=torch.uint8)  # Type conversion (not in-place)

    # import matplotlib.pyplot as plt
    # img_0 = images[0].permute([1, 2, 0])
    # plt.imshow(img_0.cpu().numpy())
    # plt.savefig('saved_image_0.png', dpi=300, bbox_inches='tight')
    # img_1 = images[1].permute([1, 2, 0])
    # plt.imshow(img_1.cpu().numpy())
    # plt.savefig('saved_image_1.png', dpi=300, bbox_inches='tight')
    # img_2 = images[2].permute([1, 2, 0])
    # plt.imshow(img_2.cpu().numpy())
    # plt.savefig('saved_image_2.png', dpi=300, bbox_inches='tight')
    # img_3 = images[3].permute([1, 2, 0])
    # plt.imshow(img_3.cpu().numpy())
    # plt.savefig('saved_image_3.png', dpi=300, bbox_inches='tight')

    return images


class pretrained_image_features(ManagerTermBase):
    """Extracted image features from a pre-trained vision encoder.

    This term extracts features from images using configurable vision encoders (R3M, CNN, etc.).
    It calls the :func:`image` function to get the images and then processes them using the selected encoder.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_device: The device to store and infer the model on. Defaults to the environment device.
        encoder_type: Type of vision encoder to use ("r3m", "cnn"). Defaults to "r3m".
        feature_dim: Dimension of output feature vector. Defaults to 512.
        encoder_config: Additional configuration for the encoder. Defaults to None.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # Initialize the base class
        super().__init__(cfg, env)

        # Extract parameters from the configuration
        self.activation = cfg.params.get("activation", "elu")
        self.image_feature_dim = cfg.params.get("image_feature_dim", None)
        self.encoder_type = cfg.params.get("encoder_type", "r3m")
        self.encoder_config = cfg.params.get("encoder_config", {"model_name": "resnet18"})
        self.input_shape = cfg.params.get("input_shape", (3, 224, 224))

        # Import the vision encoder
        from isaaclab_rl.rsl_rl.ext.modules.vision_encoder import (
            get_vision_encoder
        )

        # Initialize the encoder with the specified configuration
        self._model = get_vision_encoder(
            encoder_type=self.encoder_type,
            input_shape=self.input_shape,
            feature_dim=None,  # Directly use pretrained features
            activation=self.activation,
            freeze=True,  # Not training the encoder
            encoder_config=self.encoder_config
        ).to(env.device)

        # Print initialization info
        print(f"Initialized {self.encoder_type} vision encoder with feature dimension {self.image_feature_dim}")
        if self.encoder_config:
            print(f"Encoder config: {self.encoder_config}")

    def reset(self, env_ids: torch.Tensor | None = None):
        # No stateful components that need resetting
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        activation: str = "elu",
        image_feature_dim: Optional[int] = None,
        encoder_type: str = "r3m",
        encoder_config: Optional[Dict[str, Any]] = {"model_name": "resnet18"},
        input_shape: tuple = (3, 224, 224),
    ) -> torch.Tensor:
        """Extract image features using the configured vision encoder.

        Args:
            env: The environment to extract features from
            sensor_cfg: The sensor configuration to poll
            data_type: The sensor data type (only "rgb" supported currently)
            convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images
            **kwargs: Additional arguments (ignored)

        Returns:
            Tensor of extracted features with shape (num_envs, feature_dim)
        """
        # Obtain the images from the sensor
        # Use our custom process_image function to get images in the right format
        image_data = process_image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            process_image=True,  # Apply basic preprocessing
            output_size=input_shape[1:],  # Resize to the input shape of the encoder
        )

        # Forward the images through the model
        with torch.no_grad():
            features = self._model(image_data)

        return features


class last_joint_pos(ManagerTermBase):
    """The previous joint positions of the asset.

    This function tracks and returns the joint positions from the previous timestep.
    It's similar to last_action but for joint positions instead of actions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.

    Args:
        env: The environment to extract joint positions from
        asset_cfg: The asset configuration to track joint positions from. Defaults to SceneEntityCfg("robot").

    Returns:
        The previous joint positions tensor. Shape is (num_envs, num_joints).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # Initialize the base class
        super().__init__(cfg, env)

        # Extract parameters from the configuration
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        
        # Get the number of joints from the configuration
        joint_ids = self.asset_cfg.joint_ids
        self.num_joints = len(joint_ids)

        # Initialize the joint positions history buffer for all timesteps
        self._joint_pos_history = torch.zeros((env.num_envs, env.unwrapped.max_episode_length + 1, self.num_joints), device=env.device, dtype=torch.float32)
        self._current_timestep = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | None = None):
        # Reset joint position history to zeros for specified environments
        if env_ids is not None:
            self._joint_pos_history[env_ids] = 0.0
            self._current_timestep[env_ids] = 0
        else:
            self._joint_pos_history.zero_()
            self._current_timestep.zero_()

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Extract joint positions from the asset.

        Args:
            env: The environment to extract joint positions from
            asset_cfg: The asset configuration to track joint positions from. Defaults to SceneEntityCfg("robot").
        """
        if hasattr(env.unwrapped, 'episode_length_buf'):
            self._current_timestep = env.unwrapped.episode_length_buf.clone()
        else:
            self._current_timestep = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # Get the current joint positions
        current_robot_asset: Articulation = env.scene[self.asset_cfg.name]
        current_joint_pos = current_robot_asset.data.joint_pos[:, self.asset_cfg.joint_ids].clone().to(env.device)

        # Store current joint positions in history buffer
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        self._joint_pos_history[env_ids, self._current_timestep] = current_joint_pos

        # Get previous timestep's joint positions (t-1) with proper bounds checking
        prev_timestep = torch.clamp(self._current_timestep - 1, min=0)
        prev_joint_pos = self._joint_pos_history[env_ids, prev_timestep].clone().to(env.device)

        # Zero out values when we're at the first timestep (t=0)
        prev_joint_pos = torch.where(
            (self._current_timestep > 0).unsqueeze(1).expand(-1, self.num_joints),
            prev_joint_pos,
            torch.zeros((env.num_envs, self.num_joints), device=env.device, dtype=torch.float32)
        )

        # Return the previous joint positions
        return prev_joint_pos


def last_processed_action(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required.
    """
    return env.action_manager.get_term(action_name).processed_actions
