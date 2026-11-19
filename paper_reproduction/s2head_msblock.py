
"""S2Head Figure 4 reconstruction: RepDWUnit and MSBlock.

These are architecture components, not a complete trained detector.
MaxPool kernel=2, stride=2 is an explicit reconstruction choice.
Stride-1 residual use is restricted to matching input/output channels.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from s2head_blocks import conv_bn


def depthwise_bn(channels, kernel):
    return nn.Sequential(
        nn.Conv2d(
            channels, channels, kernel,
            stride=1, padding=kernel // 2,
            groups=channels, bias=False,
        ),
        nn.BatchNorm2d(channels),
    )


class RepDWUnit(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.pointwise = nn.Sequential(
            conv_bn(c1, c2, 1, 1),
            nn.SiLU(),
        )
        self.branch3 = depthwise_bn(c2, 3)
        self.branch7 = depthwise_bn(c2, 7)
        self.activation = nn.SiLU()
        self.deploy = None

    def forward(self, x):
        x = self.pointwise(x)
        if self.deploy is not None:
            return self.activation(self.deploy(x))
        return self.activation(self.branch3(x) + self.branch7(x))

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy is not None:
            return self
        if self.training:
            raise RuntimeError("Call eval() before fusion.")

        conv3 = fuse_conv_bn_eval(self.branch3[0], self.branch3[1])
        conv7 = fuse_conv_bn_eval(self.branch7[0], self.branch7[1])

        self.deploy = conv7
        self.deploy.weight.copy_(
            conv7.weight + F.pad(conv3.weight, (2, 2, 2, 2))
        )
        self.deploy.bias.copy_(conv7.bias + conv3.bias)

        # Fold the pointwise convolution's BN as well.
        pointwise = self.pointwise[0]
        self.pointwise[0] = fuse_conv_bn_eval(
            pointwise[0], pointwise[1]
        )

        del self.branch3
        del self.branch7
        return self


class MSBlock(nn.Module):
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("Supported strides are 1 and 2.")
        if stride == 1 and c1 != c2:
            raise ValueError(
                "Stride-1 residual requires matching channels. "
                "A channel projection is not assumed here."
            )

        self.stride = stride
        self.unit1 = RepDWUnit(c1, c2)
        self.unit2 = RepDWUnit(c2, c2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.unit2(self.unit1(x))
        if self.stride == 1:
            return x + y
        return self.pool(y)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("Call eval() before fusion.")
        self.unit1.switch_to_deploy()
        self.unit2.switch_to_deploy()
        return self


def verify():
    torch.manual_seed(42)

    cases = [
        (16, 16, 1, 32, 32),
        (16, 32, 2, 32, 32),
        (32, 32, 1, 17, 19),
        (32, 64, 2, 17, 19),
    ]

    print("Device: CPU float32")

    for c1, c2, stride, height, width in cases:
        block = MSBlock(c1, c2, stride).eval()

        with torch.no_grad():
            for layer in block.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.running_mean.uniform_(-0.5, 0.5)
                    layer.running_var.uniform_(0.5, 1.5)
                    layer.weight.uniform_(0.5, 1.5)
                    layer.bias.uniform_(-0.2, 0.2)

        x = torch.randn(2, c1, height, width)
        fused = copy.deepcopy(block).switch_to_deploy()

        with torch.no_grad():
            expected = block(x)
            actual = fused(x)

        output_height = height if stride == 1 else height // 2
        output_width = width if stride == 1 else width // 2
        assert actual.shape == (
            2, c2, output_height, output_width
        )

        torch.testing.assert_close(
            expected, actual, atol=1e-5, rtol=1e-4
        )

        # Calling fusion twice must remain safe.
        fused.switch_to_deploy()
        with torch.no_grad():
            repeated = fused(x)
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)

        block.train()
        block.zero_grad(set_to_none=True)
        block(x).square().mean().backward()

        for unit in (block.unit1, block.unit2):
            for branch in (unit.branch3, unit.branch7):
                gradient = branch[0].weight.grad
                assert gradient is not None
                assert torch.isfinite(gradient).all()
                assert gradient.abs().sum() > 0

        assert not any(
            isinstance(layer, nn.BatchNorm2d)
            for layer in fused.modules()
        )

        error = (expected - actual).abs().max().item()
        print(
            f"PASS: {c1}->{c2}, stride={stride}, "
            f"output={tuple(actual.shape)}, "
            f"fusion max error={error:.8f}; "
            "all depthwise branches receive gradients"
        )

    print("\nMSBLOCK CHECK PASS")
    print("No dataset access, training, or checkpoint changes.")


if __name__ == "__main__":
    verify()
