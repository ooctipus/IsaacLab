import torch
import torch.nn as nn

CNN_OUT_FEATURES = 32


def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function to compute the output size of a convolution layer.

    h_w: Tuple[int, int] - height and width of the input
    kernel_size: int or Tuple[int, int] - size of the convolution kernel
    stride: int or Tuple[int, int] - stride of the convolution
    pad: int or Tuple[int, int] - padding
    dilation: int or Tuple[int, int] - dilation rate
    """
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size

    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride

    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad

    h = (h_w[0] + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    w = (h_w[1] + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    return h, w


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims).contiguous()


class CustomCNN(nn.Module):
    def __init__(self, input_height, input_width, depth=True, num_channel=1):
        super().__init__()

        # Initial input dimensions
        h, w = input_height, input_width

        # Layer 1
        h, w = conv_output_size((h, w), kernel_size=6, stride=2)
        layer1_norm_shape = [16, h, w]

        # Layer 2
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer2_norm_shape = [32, h, w]

        # Layer 3
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer3_norm_shape = [64, h, w]

        # Layer 4
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer4_norm_shape = [128, h, w]

        # CNN definition
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer1_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer2_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer3_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer4_norm_shape),  # Dynamically calculated layer norm
            nn.AdaptiveAvgPool2d((1, 1))  # Pool to (1, 1) feature map for any input size
        )

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(128, CNN_OUT_FEATURES)
        )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128))
        return out
