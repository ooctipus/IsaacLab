from typing import Literal
from isaaclab.utils import configclass
from .encoder_networks import CNN, MLP


@configclass
class CNNCfg:

    class_type: type[CNN] = CNN

    channels: list[int] = [32, 64, 128]

    kernel_sizes: list[int] = [3, 3, 3]

    strides: list[int] = [2, 2, 2]

    paddings: list[int] = [1, 1, 1]

    activation: Literal["elu", "selu", "relu", "crelu", "lrelu", "tanh", "sigmoid", "identity"] = "elu"

    norm: Literal["batch", "layer", "group"] | None = None

    use_maxpool: bool = True

    pool_size: int = 2

    gap: bool = True

    feature_size: int | None = None

    permute: bool = False

    input_norm: bool = False


@configclass
class MLPCfg:
    class_type: type[MLP] = MLP

    layers: list[int] = [512, 256, 128]

    activation: Literal["elu", "selu", "relu", "crelu", "lrelu", "tanh", "sigmoid", "identity"] = "elu"

    norm: Literal["batch", "layer", "group"] | None = None

    dropout: float | None = None

    feature_size: int | None = None

    input_norm: bool = False
