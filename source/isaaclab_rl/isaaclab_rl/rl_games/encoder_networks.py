import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims).contiguous()


class CNN(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], *args, **kwargs):
        super().__init__()
        channels = kwargs["channels"]
        kernel_sizes = kwargs["kernel_sizes"]
        strides = kwargs["strides"]
        paddings = kwargs["paddings"]
        activation = kwargs.get("activation", "relu")
        norm = kwargs.get("norm", "batch")
        use_maxpool = kwargs.get("use_maxpool", False)
        pool_size = kwargs.get("pool_size", 2)
        gap = kwargs.get("gap", True)
        feature_size = kwargs.get("feature_size", None)
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings), "lists must match length"
        act_cls = {
            "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "none": nn.Identity,
        }[activation.lower()]

        C, H, W = input_shape
        layers: list[nn.Module] = []
        in_c = C
        for i, out_c in enumerate(channels):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            if norm == "batch":
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(act_cls())
            if use_maxpool:
                layers.append(nn.MaxPool2d(pool_size))
            in_c = out_c

        if gap:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # -> (B, C, 1, 1)

        self.encoder = nn.Sequential(*layers)

        # figure out flattened dim with a dummy forward
        with torch.no_grad():
            # match dtype/device of current params (likely CPU here, but robust if moved)
            p = next(self.encoder.parameters(), None)
            dummy = torch.zeros(1, C, H, W, device=p.device)
            enc = self.encoder(dummy)
            flat_dim = enc.shape[1] if gap else enc.view(1, -1).shape[1]

        self.flatten = (lambda t: t.view(t.size(0), -1)) if not gap else (lambda t: t.view(t.size(0), t.size(1)))
        if feature_size is None:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(flat_dim, feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)  # expect x as (B, C, H, W)
        y = self.flatten(y)  # (B, flat_dim)
        return self.projector(y)  # (B, out_dim)
