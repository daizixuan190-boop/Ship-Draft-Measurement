import torch
import torch.nn as nn


class GAB(nn.Module):
    """Geometry Adaptation Block used in the final MD-YOLO release."""

    def __init__(self, in_planes, out_planes=None, mode=None, kernel_global=7, stride=1):
        super().__init__()

        # Compatibility for legacy YAML variants:
        # GAB(c1, c2), GAB(c1, c2, "semantic"), GAB(c1, c2, 7), GAB(c1, c2, 7, 1)
        if isinstance(mode, (int, float)) and isinstance(kernel_global, (int, float)):
            stride = int(kernel_global)
            kernel_global = int(mode)
            mode = None
        elif isinstance(mode, (int, float)):
            kernel_global = int(mode)
            mode = None
        elif isinstance(kernel_global, str):
            mode = kernel_global
            kernel_global = 7

        out_planes = in_planes if out_planes is None else out_planes
        self.mode = mode
        gate_channels = max(in_planes // 4, 1)

        self.shortcut = (
            nn.Identity()
            if in_planes == out_planes and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        )

        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, gate_channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(gate_channels, 2, 1, bias=False),
            nn.Sigmoid(),
        )

        self.local_path = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, stride, padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.SiLU(),
            nn.Conv2d(in_planes, out_planes, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.SiLU(),
        )

        self.global_path = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.Conv2d(
                out_planes,
                out_planes,
                (kernel_global, 1),
                1,
                (kernel_global // 2, 0),
                groups=out_planes,
                bias=False,
            ),
            nn.Conv2d(
                out_planes,
                out_planes,
                (1, kernel_global),
                1,
                (0, kernel_global // 2),
                groups=out_planes,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.SiLU(),
        )

    def forward(self, x):
        identity = self.shortcut(x)
        gains = self.dynamic_gate(x)
        alpha = gains[:, 0:1]
        beta = gains[:, 1:2]
        out_local = self.local_path(x)
        out_global = self.global_path(x)
        return identity + alpha * out_local + beta * out_global
