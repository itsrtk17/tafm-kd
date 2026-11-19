
"""Weighted concatenation primitive for the S2Head reconstruction.

Figure 2 specifies wConcat nodes.
ReLU weight normalization and epsilon=1e-4 are explicit
implementation choices, not a formula specified by the paper.

Inputs must already have matching spatial dimensions.
This module does not resize or project features.
"""

import torch
from torch import nn


class WeightedConcat(nn.Module):
    def __init__(self, inputs, eps=1e-4):
        super().__init__()
        if inputs < 2:
            raise ValueError("At least two inputs are required.")
        if eps <= 0:
            raise ValueError("epsilon must be positive.")

        self.weights = nn.Parameter(torch.ones(inputs))
        self.eps = eps

    def forward(self, features):
        if len(features) != self.weights.numel():
            raise ValueError("Unexpected number of feature maps.")

        reference = features[0]
        if reference.ndim != 4:
            raise ValueError("Expected BCHW feature maps.")

        for feature in features:
            if feature.ndim != 4:
                raise ValueError("Expected BCHW feature maps.")
            if (
                feature.shape[0] != reference.shape[0]
                or feature.shape[2:] != reference.shape[2:]
            ):
                raise ValueError(
                    "Align batch and spatial dimensions before fusion."
                )
            if (
                feature.device != reference.device
                or feature.dtype != reference.dtype
            ):
                raise ValueError("Input devices and dtypes must match.")

        positive = self.weights.relu()
        normalized = positive / (positive.sum() + self.eps)

        return torch.cat([
            feature * normalized[index]
            for index, feature in enumerate(features)
        ], dim=1)


def verify():
    torch.manual_seed(42)

    for channels in [(16, 32), (16, 32, 64)]:
        node = WeightedConcat(len(channels))
        features = [
            torch.randn(2, channel, 16, 16, requires_grad=True)
            for channel in channels
        ]

        with torch.no_grad():
            node.weights.copy_(
                torch.arange(1, len(channels) + 1).float()
            )

        output = node(features)
        assert output.shape == (2, sum(channels), 16, 16)

        # Independently check scaling and channel order.
        offset = 0
        denominator = (
            sum(range(1, len(channels) + 1)) + node.eps
        )
        for index, feature in enumerate(features):
            width = feature.shape[1]
            expected = feature * ((index + 1) / denominator)
            torch.testing.assert_close(
                output[:, offset:offset + width], expected
            )
            offset += width

        output.square().mean().backward()

        assert node.weights.grad is not None
        assert torch.isfinite(node.weights.grad).all()
        assert (node.weights.grad.abs() > 0).all()

        for feature in features:
            assert feature.grad is not None
            assert torch.isfinite(feature.grad).all()
            assert feature.grad.abs().sum() > 0

        # Degenerate weights must not cause division by zero.
        with torch.no_grad():
            node.weights.fill_(-1)
            zero_output = node([f.detach() for f in features])
            assert torch.isfinite(zero_output).all()
            assert torch.count_nonzero(zero_output) == 0

        print(
            f"PASS: {len(channels)} inputs, "
            f"channels={channels}, output={tuple(output.shape)}; "
            "scaling, channel order and gradients verified"
        )

    node = WeightedConcat(2)
    try:
        node([
            torch.randn(1, 16, 16, 16),
            torch.randn(1, 16, 8, 8),
        ])
    except ValueError:
        print("PASS: spatial mismatch rejected")
    else:
        raise AssertionError("Spatial mismatch was not rejected.")

    print("\nWEIGHTED CONCAT CHECK PASS")
    print("No dataset access, training, or checkpoint changes.")


if __name__ == "__main__":
    verify()
