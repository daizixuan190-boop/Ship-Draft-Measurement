import torch
import torch.nn as nn

__all__ = ["MobileNetV4ConvSmall", "MobileNetV4ConvSmall_Slim", "MNV4FeatureSelect"]


MNV4ConvSmall_Slim_SPECS = {
    "conv0": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 32, 3, 2]]},
    "layer1": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[32, 32, 3, 2], [32, 32, 1, 1]]},
    "layer2": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[32, 96, 3, 2], [96, 64, 1, 1]]},
    "layer3": {
        "block_name": "uib",
        "num_blocks": 3,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 3, 0, True, 1, 4],
        ],
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 3,
        "block_specs": [
            [96, 128, 3, 3, True, 2, 6],
            [128, 128, 5, 5, True, 1, 4],
            [128, 128, 0, 3, True, 1, 4],
        ],
    },
}


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    padding = (kernel_size - 1) // 2
    layers = [nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups)]
    if norm:
        layers.append(nn.BatchNorm2d(oup))
    if act:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        start_dw_kernel_size,
        middle_dw_kernel_size,
        middle_dw_downsample,
        stride,
        expand_ratio,
    ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.use_res_connect = stride == 1 and inp == oup

        if self.start_dw_kernel_size > 0:
            stride_ = stride if not middle_dw_downsample else 1
            self.start_dw = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)

        expand_filters = make_divisible(inp * expand_ratio, 8)
        self.expand_conv = conv_2d(inp, expand_filters, kernel_size=1)

        if self.middle_dw_kernel_size > 0:
            stride_ = stride if middle_dw_downsample else 1
            self.middle_dw = conv_2d(
                expand_filters,
                expand_filters,
                kernel_size=middle_dw_kernel_size,
                stride=stride_,
                groups=expand_filters,
            )

        self.proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        shortcut = x
        if self.start_dw_kernel_size > 0:
            x = self.start_dw(x)
        x = self.expand_conv(x)
        if self.middle_dw_kernel_size > 0:
            x = self.middle_dw(x)
        x = self.proj_conv(x)
        return x + shortcut if self.use_res_connect else x


def build_blocks(layer_spec):
    if not layer_spec:
        return nn.Sequential()

    layers = []
    if layer_spec["block_name"] == "convbn":
        schema = ["inp", "oup", "kernel_size", "stride"]
        for spec in layer_spec["block_specs"]:
            args = dict(zip(schema, spec))
            layers.append(conv_2d(**args))
    elif layer_spec["block_name"] == "uib":
        schema = [
            "inp",
            "oup",
            "start_dw_kernel_size",
            "middle_dw_kernel_size",
            "middle_dw_downsample",
            "stride",
            "expand_ratio",
        ]
        for spec in layer_spec["block_specs"]:
            args = dict(zip(schema, spec))
            layers.append(UniversalInvertedBottleneckBlock(**args))
    return nn.Sequential(*layers)


class MobileNetV4(nn.Module):
    """Reduced MobileNetV4-Small backbone used by the final MD-YOLO model."""

    def __init__(self, is_slim=True):
        super().__init__()
        self.spec = MNV4ConvSmall_Slim_SPECS
        self.conv0 = build_blocks(self.spec["conv0"])
        self.layer1 = build_blocks(self.spec["layer1"])
        self.layer2 = build_blocks(self.spec["layer2"])
        self.layer3 = build_blocks(self.spec["layer3"])
        self.layer4 = build_blocks(self.spec["layer4"])
        self.channel = [64, 96, 128]

    def forward(self, x):
        x = self.conv0(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return [p3, p4, p5]


class MNV4FeatureSelect(nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


def MobileNetV4ConvSmall():
    return MobileNetV4(is_slim=False)


def MobileNetV4ConvSmall_Slim():
    return MobileNetV4(is_slim=True)
