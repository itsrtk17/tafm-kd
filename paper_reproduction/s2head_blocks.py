
"""S2Head components reconstructed from reference [34].

This file currently implements RepBlock only.
It is not yet a complete S2Head or TAFM-KD implementation.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


def conv_bn(c1, c2, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(
            c1, c2, kernel_size=kernel,
            stride=stride, padding=kernel // 2,
            bias=False,
        ),
        nn.BatchNorm2d(c2),
    )


class RepBlock(nn.Module):
    """Parallel 3x3/1x1 branches, summed before activation.

    SiLU is an explicit reconstruction choice pending comparison
    with the architecture diagrams or author implementation.
    """

    def __init__(self, c1, c2, stride=2):
        super().__init__()
        self.branch3 = conv_bn(c1, c2, 3, stride)
        self.branch1 = conv_bn(c1, c2, 1, stride)
        self.activation = nn.SiLU()
        self.deploy = None

    def forward(self, x):
        if self.deploy is not None:
            return self.activation(self.deploy(x))
        return self.activation(self.branch3(x) + self.branch1(x))

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy is not None:
            return self

        if self.training:
            raise RuntimeError(
                "Call eval() before combining convolution/BN branches."
            )

        conv3 = fuse_conv_bn_eval(self.branch3[0], self.branch3[1])
        conv1 = fuse_conv_bn_eval(self.branch1[0], self.branch1[1])

        self.deploy = conv3
        self.deploy.weight.copy_(
            conv3.weight + F.pad(conv1.weight, (1, 1, 1, 1))
        )
        self.deploy.bias.copy_(conv3.bias + conv1.bias)

        del self.branch3
        del self.branch1
        return self


def verify():
    torch.manual_seed(42)

    cases = [
        (3, 16, 2, 64, 64),
        (16, 32, 2, 65, 63),
        (32, 32, 1, 32, 32),
    ]

    print("Device: CPU float32")
    for c1, c2, stride, height, width in cases:
        block = RepBlock(c1, c2, stride).eval()

        # Nontrivial BN statistics test the fusion formula properly.
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
            before = block(x)
            after = fused(x)

        torch.testing.assert_close(
            before, after, atol=1e-5, rtol=1e-4
        )

        # Check that the training structure remains differentiable.
        block.train()
        block.zero_grad(set_to_none=True)
        loss = block(x).square().mean()
        loss.backward()

        for branch in (block.branch3, block.branch1):
            gradient = branch[0].weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0

        error = (before - after).abs().max().item()
        print(
            f"PASS: channels {c1}->{c2}, stride={stride}, "
            f"shape={tuple(before.shape)}, "
            f"fusion max error={error:.8f}; "
            "both branches receive gradients"
        )

    print("\nREPBLOCK CHECK PASS")
    print("No dataset access, training run, or checkpoint changes.")


if __name__ == "__main__":
    verify()
