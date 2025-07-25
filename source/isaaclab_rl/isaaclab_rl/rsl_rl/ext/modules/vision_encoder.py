import torch
import numpy as np
import torch.nn as nn
from gymnasium import spaces
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Tuple, Dict, List, Optional, Union, Any
from enum import Enum

# Import the activation resolver
from rsl_rl.utils import resolve_nn_activation


class EncoderType(str, Enum):
    """Enum for supported encoder types."""
    CNN = "cnn"
    R3M = "r3m"
    DINOv2 = "dinov2"


class VisionEncoder(nn.Module):
    """Base class for all vision encoders."""
    def __init__(
        self,
        obs_space: spaces.Box,
        feature_dim: Optional[int] = 128,
        activation: str = "relu",
        freeze: bool = True,
        normalize: bool = True,
        normalize_style: str = "normal"
    ):
        super().__init__()
        self.obs_space = obs_space
        self.feature_dim = feature_dim
        self.activation_fn = resolve_nn_activation(activation)
        self.freeze_encoder = freeze
        self.normalize_style = normalize_style
        self._processors: List[Callable] = []
        self._processor_descriptions: List[str] = []
        
        if obs_space.shape[1] in [3, 1, 4]:
            self.num_channel = obs_space.shape[1]
        elif obs_space.shape[-1] in [3, 1, 4]:
            self._processors.append(lambda x : x.permute(0, 3, 1, 2))
            self._processor_descriptions.append("permute HWC->CHW")
            self.num_channel = obs_space.shape[-1]
        else:
            raise ValueError("did not detect correct channel")

        if obs_space.dtype != np.float32:
            self._processors.append(lambda x : x.float())
            self._processor_descriptions.append("cast to float32")
        if normalize:  # rgb indicator
            if obs_space.shape[1] == 3 or obs_space.shape[-1] == 3:
                processors, descriptions = self._compile_rgb_processors(obs_space)
                self._processors.extend(processors)
                self._processor_descriptions.extend(descriptions)
            elif obs_space.shape[1] == 1 or obs_space.shape[-1] == 1:
                processors, descriptions = self._compile_depth_processors(obs_space)
                self._processors.extend(processors)
                self._processor_descriptions.extend(descriptions)
            elif obs_space.shape[1] == 4 or obs_space.shape[-1] == 4:
                processors, descriptions = self._compile_rgbd_processors(obs_space)
                self._processors.extend(processors)
                self._processor_descriptions.extend(descriptions)
    def freeze(self):
        """Freeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def _compile_rgb_processors(self, obs_space: Any) -> List[Any]:
        """Build processors for 3-channel inputs."""
        procs: List[callable] = []
        desc: List[str] = []
        if self.normalize_style == "imagenet":
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
        elif self.normalize_style == "normal":
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
        if self.feature_dim:
            return nn.Sequential(
                nn.Linear(input_dim, self.feature_dim),
                self.activation_fn
            )
        else:
            return nn.Identity()

    def initialize(self):
        """Hook called at the end of initialization to set up model."""
        if self.freeze_encoder:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        Args:
            x: Input tensor in shape (B, C, H, W)
        Returns:
            Encoded features in shape (B, feature_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class CNNEncoder(VisionEncoder):
    """CNN encoder with configurable architecture."""

    def __init__(
        self,
        obs_space: Any,                      # e.g. gym.spaces.Box
        feature_dim: Optional[int] = 128,
        activation: str = "relu",
        freeze: bool = False,
        normalize: bool = True,
        normalize_style: str = "normal",
        config: Optional[Dict[str, Any]] = None,
    ):
        # 1) Initialize base VisionEncoder, which sets up self._processors + normalization
        super().__init__(
            obs_space,
            feature_dim=feature_dim,
            activation=activation,
            freeze=freeze,
            normalize=normalize,
            normalize_style=normalize_style,
        )
        assert not freeze, "CNNEncoder: freezing not supported for untrained CNN"

        # 2) CNN architecture config
        self.config = {
            "channels": [32, 64, 128],
            "kernel_sizes":[3, 3, 3],
            "strides": [2, 2, 2],
            "paddings": [1, 1, 1],
            "use_maxpool": True,
            "pool_size": 2,
        }
        if config is not None:
            self.config.update(config)

        # 3) Build the conv / flatten stack and the projector
        self._build_encoder()
        self.initialize()

    def _build_encoder(self):
        layers: List[nn.Module] = []
        in_ch = self.num_channel
        # conv → activation → optional pool
        for i, out_ch in enumerate(self.config["channels"]):
            layers.append(nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=self.config["kernel_sizes"][i],
                stride=self.config["strides"][i],
                padding=self.config["paddings"][i],
            ))
            layers.append(self.activation_fn)
            if self.config["use_maxpool"]:
                layers.append(nn.MaxPool2d(self.config["pool_size"]))
            in_ch = out_ch

        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

        # determine flattened size via a dummy forward (after preprocessing)
        with torch.no_grad():
            dummy = torch.zeros(1, *self.obs_space.shape[1:])
            dummy = self.preprocess(dummy)          # HWC→CHW, cast, normalize
            flat_sz = self.encoder(dummy).shape[1]

        self.projector = self.build_projector(flat_sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x may be uint8 [0 - 255] or float, HWC or CHW.
        VisionEncoder.preprocess will handle all casts, permutes, and norms.
        """
        x = self.preprocess(x)      # now BCHW, float, normalized
        feats = self.encoder(x)     # → (B, flat_dim)
        return self.projector(feats)  # → (B, feature_dim)


class R3MEncoder(VisionEncoder):
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
    
    def _ensure_chw(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.input_shape[0]:
            return x.permute(0, 3, 1, 2)
        return x

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


class DINOv2Encoder(VisionEncoder):
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


def get_vision_encoder(
    encoder_type: Union[str, EncoderType],
    observation_space: any,
    feature_dim: Optional[int] = 128,
    activation: str = "relu",
    freeze: bool = True,
    encoder_config: Optional[Dict[str, Any]] = None,
    normalize = True
) -> VisionEncoder:
    """
    Factory function to create a vision encoder based on type.
    Args:
        encoder_type: Type of encoder to create
        input_shape: Shape of input images (C, H, W)
        feature_dim: Dimension of output feature vector
        activation: Activation function to use
        freeze: Whether to freeze the encoder parameters
        encoder_config: Additional arguments specific to certain encoders
    Returns:
        VisionEncoder: An instance of the requested encoder type
    """
    # Convert string to enum if needed
    if isinstance(encoder_type, str):
        try:
            encoder_type = EncoderType(encoder_type.lower())
        except ValueError:
            valid_types = [e.value for e in EncoderType]
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. "
                f"Available types: {valid_types}"
            )

    # Create appropriate encoder
    if encoder_type == EncoderType.CNN:
        return CNNEncoder(
            obs_space=observation_space,
            feature_dim=feature_dim,
            activation=activation,
            freeze=freeze,
            config=encoder_config,
            normalize=normalize
        )
    elif encoder_type == EncoderType.R3M:
        return R3MEncoder(
            input_shape=observation_space,
            feature_dim=feature_dim,
            activation=activation,
            freeze=freeze,
            config=encoder_config
        )
    elif encoder_type == EncoderType.DINOv2:
        return DINOv2Encoder(
            input_shape=observation_space,
            feature_dim=feature_dim,
            activation=activation,
            freeze=freeze,
            config=encoder_config
        )
    else:
        raise ValueError(f"Encoder type {encoder_type} is not implemented")
