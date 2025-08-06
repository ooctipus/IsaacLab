from __future__ import annotations
import torch
import numpy as np
import torch.nn as nn
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional, Any, TYPE_CHECKING

# Import the activation resolver
from rsl_rl.utils import resolve_nn_activation

if TYPE_CHECKING:
    from ...actor_critic_vision_cfg import CNNEncoderCfg, PretrainedEncoderCfg, PointNetEncoderCfg, ActorCriticVisionAdapterCfg

class VisionAdapter(nn.Module):
    """Base class for all vision encoders."""
    def __init__(self, obs_space: spaces.Box, adapter_cfg: ActorCriticVisionAdapterCfg):
        super().__init__()
        self.cfg = adapter_cfg
        self.obs_space = obs_space
        self._processors: List[callable] = []
        self._processor_descriptions: List[str] = []

        if obs_space.dtype != np.float32:
            self._processors.append(lambda x : x.float())
            self._processor_descriptions.append("cast to float32")
        
        if len(obs_space.shape) == 4:
            # determine whether permutation processor and dtype caster processor is needed
            # is normalize is false, permutation and dtype casting should be only processing needed
            if obs_space.shape[1] in [3, 1, 4]:
                self.num_channel = obs_space.shape[1]
            elif obs_space.shape[-1] in [3, 1, 4]:
                self._processors.append(lambda x : x.permute(0, 3, 1, 2))
                self._processor_descriptions.append("permute HWC->CHW")
                self.num_channel = obs_space.shape[-1]
            else:
                raise ValueError("did not detect correct channel")
            
            # if normalize is True we need more processors.
            if self.cfg.normalize:
                if obs_space.shape[1] == 3 or obs_space.shape[-1] == 3:  # rgb indicator
                    processors, descriptions = self._compile_rgb_processors(obs_space)
                    self._processors.extend(processors)
                    self._processor_descriptions.extend(descriptions)
                elif obs_space.shape[1] == 1 or obs_space.shape[-1] == 1:  # depth indicator
                    processors, descriptions = self._compile_depth_processors(obs_space)
                    self._processors.extend(processors)
                    self._processor_descriptions.extend(descriptions)
                elif obs_space.shape[1] == 4 or obs_space.shape[-1] == 4:  # rgbd indicator
                    processors, descriptions = self._compile_rgbd_processors(obs_space)
                    self._processors.extend(processors)
                    self._processor_descriptions.extend(descriptions)
        elif len(obs_space.shape) == 3:
            # expect (B, N_points, C)
            self.num_channel = obs_space.shape[-1]
            if self.cfg.normalize:
                procs, desc = self._compile_point_cloud_processors(obs_space)
                self._processors.extend(procs)
                self._processor_descriptions.extend(desc)
        else:
            raise ValueError(f"Does not recognize this perception with dim {len(obs_space.shape)}")

    def freeze(self):
        """Freeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def _compile_point_cloud_processors(self, obs_space: Any) -> Tuple[List[callable], List[str]]:
        """Build processors for point-cloud inputs (center + unit-sphere scale)."""
        procs: List[callable] = []
        desc: List[str] = []
        # center each cloud at its centroid
        procs.append(lambda x: x - x.mean(dim=1, keepdim=True))
        desc.append("center to centroid")
        # scale to fit in unit sphere
        def scale_unit(x):
            d = torch.norm(x, dim=-1)            # (B, N)
            m = d.max(dim=1, keepdim=True)[0]    # (B, 1)
            return x / (m.unsqueeze(-1) + 1e-6)
        procs.append(scale_unit)
        desc.append("scale to unit sphere")
        return procs, desc
    
    def _compile_rgb_processors(self, obs_space: Any) -> List[Any]:
        """Build processors for 3-channel inputs."""
        procs: List[callable] = []
        desc: List[str] = []
        if self.cfg.normalize_style == "imagenet":
            # ImageNet stats
            mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            if obs_space.high == 255:
                procs.append(lambda x: x/255.0)
                desc.append("rgb: scale / 255")
                procs.append(lambda x: (x-mean)/std)
                desc.append("ImageNet normalize")
            else:
                procs.append(lambda x: (x-mean)/std)
                desc.append("ImageNet normalize")
        elif self.cfg.normalize_style == "normal":
            # "normal": scale to [-1,1] then standardize
            if obs_space.high == 255:
                procs.append(lambda x: (x / 255.0) * 2 - 1)
                desc.append("rgb: scale [0 - 255]→[-1, 1]")
            procs.append(lambda x: (x - x.mean())/x.std())
            desc.append("rgb: per-image standardize")
        return procs, desc
    
    def _compile_depth_processors(self, obs_space: Any) -> List[Any]:
        """Build processors for single-channel depth inputs."""
        procs: List[callable] = []
        desc: List[str] = []
        high = obs_space.high
        if np.all(high == np.inf):
            procs.append(lambda x: torch.tanh(x / 2.0) * 2 - 1)
            desc.append("depth: scale [0 - inf]→[-1, 1]")
        elif np.all(high == 255.0):
            procs.append(lambda x: (x / 255.0) * 2 - 1)
            desc.append("depth: scale [0 - 255]→[-1, 1]")
        elif np.all(high == 1.0) and np.all(obs_space.low == 0.0):
            procs.append(lambda x: x * 2 - 1)
            desc.append("depth: scale [0 - 1]→[-1, 1]")
        elif np.all(high == 1.0) and np.all(obs_space.low == -1.0):
            pass # no need to do anything if already -1 to 1
        else:
            raise ValueError("Your depth image is not qualified for automatic normalization")
        procs.append(lambda x: (x - x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)) / (x.std(dim=tuple(range(1, x.ndim)), keepdim=True) + 1e-8))
        desc.append("depth: per-image standardize")
        return procs, desc

    def _compile_rgbd_processors(self, obs_space: spaces.Box):
        raise NotImplementedError()
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all preprocessing steps in sequence."""
        for fn in self._processors:
            x = fn(x)
        return x
    
    def build_projector(self, input_dim: int) -> nn.Module:
        """Create a projector that maps encoder output to desired feature dimension.
        Args:
            input_dim: Input dimension to the projector
        Returns:
            Projector module
        """
        if self.cfg.feature_dim:
            return nn.Sequential(nn.Linear(input_dim, self.cfg.feature_dim), resolve_nn_activation(self.cfg.activation))
        else:
            return nn.Identity()

    def initialize(self):
        """Hook called at the end of initialization to set up model."""
        if self.cfg.freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        Args:
            x: Input tensor in shape (B, C, H, W)
        Returns:
            Encoded features in shape (B, feature_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class CNNEncoder(VisionAdapter):
    """CNN encoder with configurable architecture."""
    
    cfg: ActorCriticVisionAdapterCfg

    def __init__(self, obs_space: spaces.Box, adapter_cfg: ActorCriticVisionAdapterCfg):
        super().__init__(obs_space, adapter_cfg)
        self._build_encoder(obs_space)
        self.initialize()

    def _build_encoder(self, obs_space):
        layers: List[nn.Module] = []
        in_c = self.num_channel
        ec: CNNEncoderCfg = self.cfg.encoder_cfg
        for i, out_c in enumerate(ec.channels):
            c = nn.Conv2d(in_c, out_c, kernel_size=ec.kernel_sizes[i], stride=ec.strides[i], padding=ec.paddings[i])
            layers.append(c)
            layers.append(resolve_nn_activation(self.cfg.activation))
            if ec.use_maxpool:
                layers.append(nn.MaxPool2d(ec.pool_size))
            in_c = out_c

        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

        # determine flattened size via a dummy forward (after preprocessing)
        with torch.no_grad():
            dummy = self.preprocess(torch.tensor(obs_space.sample())) # HWC→CHW, cast, normalize
            flat_sz = self.encoder(dummy).shape[1]

        self.projector = self.build_projector(flat_sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(self.preprocess(x))
        return self.projector(feats)  # → (B, feature_dim)


class PointNetEncoder(VisionAdapter):
    """Point-cloud encoder: per-point MLP → global-pool → projector."""
    def __init__(self, obs_space: spaces.Box, adapter_cfg: ActorCriticVisionAdapterCfg):
        super().__init__(obs_space, adapter_cfg)
        self._build_encoder(obs_space)
        self.initialize()

    def _build_encoder(self, obs_space: spaces.Box):
        pc_cfg: PointNetEncoderCfg = self.cfg.encoder_cfg
        in_c = self.num_channel
        convs: List[nn.Module] = []
        for i, out_c in enumerate(pc_cfg.channels):
            convs.append(nn.Conv1d(in_c, out_c, kernel_size=1, stride=pc_cfg.strides[i]))
            convs.append(resolve_nn_activation(self.cfg.activation))
            in_c = out_c
        self.point_mlp = nn.Sequential(*convs)

        # pooling
        if pc_cfg.use_global_feat:
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = None
            
        if pc_cfg.feature_dim is not None:
            feat_dim = pc_cfg.channels[-1]
            # projector to feature_dim
            self.projector = self.build_projector(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)            # (B,N,C)
        x = x.transpose(1,2)              # (B,C,N)
        x = self.point_mlp(x)             # (B,hidden,N)
        if self.pool:
            x = self.pool(x).squeeze(-1)  # (B, hidden)
        else:
            B,C,N = x.shape
            x = x.view(B, C*N)
        return self.projector(x)          # (B, feature_dim)

class R3MEncoder(VisionAdapter):
    """R3M pretrained encoder using ResNet backbone."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        feature_dim: Optional[int] = 64,
        activation: str = "relu",
        freeze: bool = True,
        normalize: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            input_shape: Shape of input images (C, H, W)
            feature_dim: Dimension of output feature vector
            activation: Activation function to use
            freeze: Whether to freeze the encoder parameters
            config: Configuration for R3M encoder
        """
        super().__init__(input_shape, feature_dim, activation, freeze)
        
        # Set up R3M encoder
        self.model_name = config.get("model_name", "resnet18")
        self._build_encoder()
        self.initialize()

    def _build_encoder(self):
        """Set up the R3M encoder with proper weights."""
        # Map model names to functions and output dimensions
        resnet_models = {
            "resnet18": (models.resnet18, 512),
            "resnet34": (models.resnet34, 512),
            "resnet50": (models.resnet50, 2048)
        }

        if self.model_name not in resnet_models:
            raise ValueError(
                f"Unsupported R3M backbone: {self.model_name}. "
                f"Choose from: {list(resnet_models.keys())}"
            )

        # Get model function and output dimension
        model_fn, output_dim = resnet_models[self.model_name]

        # Initialize backbone
        self.encoder = model_fn(weights=None)
        self.encoder.fc = nn.Identity()  # Remove classification head

        # Load R3M weights
        self._load_r3m_weights()

        # Create projector
        self.projector = self.build_projector(output_dim)

    def _load_r3m_weights(self):
        """Load R3M weights for the encoder."""
        try:
            # Construct model name for R3M weights
            r3m_model_name = f"r3m_{self.model_name.replace('resnet', '')}"

            # R3M weight URL pattern
            url = f"https://pytorch.s3.amazonaws.com/models/rl/r3m/{r3m_model_name}.pt"

            # Load the R3M weights
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

            # R3M weights have a specific structure with nesting
            from tensordict import TensorDict
            td = TensorDict(state_dict["r3m"], []).unflatten_keys(".")
            td_flatten = td["module"]["convnet"].flatten_keys(".")
            model_state_dict = td_flatten.to_dict()

            self.encoder.load_state_dict(model_state_dict)
            print(f"Successfully loaded R3M weights for {self.model_name}")
        except Exception as e:
            print(f"Error loading R3M weights: {e}")
            print("Using random initialization instead")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the R3M encoder.
        Args:
            x: Input tensor in shape (B, C, H, W) with values in range [0, 255]
        Returns:
            Features in shape (B, feature_dim)
        """
        # Preprocess input for ImageNet-trained models
        x = self.preprocess(x)

        # Run through encoder with gradient context depending on mode
        if not self.training:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        return self.projector(features)


class DINOv2Encoder(VisionAdapter):
    """DINOv2 pretrained encoder using ViT backbone."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        feature_dim: Optional[int] = 64,
        activation: str = "relu",
        freeze: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            input_shape: Shape of input images (C, H, W)
            feature_dim: Dimension of output feature vector
            activation: Activation function to use
            freeze: Whether to freeze the encoder parameters
            config: Configuration for DINOv2 encoder
        """
        super().__init__(input_shape, feature_dim, activation, freeze)

        # Set up DINOv2 encoder
        self.model_name = config.get("model_name", "dinov2_vitb14")
        self._build_encoder()
        self.initialize()

    def _build_encoder(self):
        """Set up the DINOv2 encoder with proper weights."""
        # Map model names to output dimensions
        dinov2_models = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536
        }

        if self.model_name not in dinov2_models:
            raise ValueError(
                f"Unsupported DINOv2 model: {self.model_name}. "
                f"Choose from: {list(dinov2_models.keys())}"
            )

        # Get output dimension
        output_dim = dinov2_models[self.model_name]

        # Initialize backbone
        self.encoder = torch.hub.load('facebookresearch/dinov2', self.model_name)

        # Create projector
        self.projector = self.build_projector(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DINOv2 encoder.
        Args:
            x: Input tensor in shape (B, C, H, W) with values in range [0, 255]
        Returns:
            Features in shape (B, feature_dim)
        """
        # Preprocess input for ImageNet-trained models
        x = self.preprocess(x)

        # Run through encoder with gradient context depending on mode
        if not self.training:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        return self.projector(features)
