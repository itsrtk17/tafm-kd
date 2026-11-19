
"""S2Head reconstruction with explicitly recorded assumptions.

Source: S2Head paper, Figures 2-5 and Section III.

Not an author-verified architecture or trained detector.
Returns raw predictions; decoding, native loss and trainer
integration are subsequent steps.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules import C2f, SPPF

from s2head_blocks import RepBlock, conv_bn
from s2head_msblock import RepDWUnit
from s2head_fusion import WeightedConcat


ASSUMPTIONS = {
    "prediction_biases": "installed Ultralytics Detect.bias_init",
    "backbone_channels": [16, 32, 64, 128, 256],
    "backbone_C2f_repeats": [1, 2, 2],
    "neck_C2f_repeats": 1,
    "C2f_expansion": 0.5,
    "upsampling": "nearest",
    "reg_max": 16,
    "fusion": "ReLU-normalized weighted concatenation",
    "MSBlock_channel_change": "first pointwise convolution; no residual",
    "MSBlock_pool": "kernel=2, stride=2",
    "head_hidden_channels": "preserve each input's channel count",
}


def cbs(c1, c2, kernel=3, stride=1):
    return nn.Sequential(
        conv_bn(c1, c2, kernel, stride),
        nn.SiLU(),
    )


class TransitionMSBlock(nn.Module):
    """Allows channel changes without inventing a residual projection."""

    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.units = nn.Sequential(
            RepDWUnit(c1, c2),
            RepDWUnit(c2, c2),
        )
        self.residual = stride == 1 and c1 == c2
        self.pool = (
            nn.MaxPool2d(2, 2) if stride == 2
            else nn.Identity()
        )

    def forward(self, x):
        y = self.units(x)
        if self.residual:
            y = y + x
        return self.pool(y)


class SlimRawHead(nn.Module):
    """Two MSBlocks per task branch, then a 1x1 prediction layer."""

    def __init__(self, channels, nc=1, reg_max=16):
        super().__init__()
        self.box = nn.Sequential(
            TransitionMSBlock(channels, channels),
            TransitionMSBlock(channels, channels),
            nn.Conv2d(channels, 4 * reg_max, 1),
        )
        self.cls = nn.Sequential(
            TransitionMSBlock(channels, channels),
            TransitionMSBlock(channels, channels),
            nn.Conv2d(channels, nc, 1),
        )

    def forward(self, x):
        return self.box(x), self.cls(x)


class S2HeadReconstruction(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = cbs(3, 16, 3, 2)
        self.stage2 = nn.Sequential(
            RepBlock(16, 32, 2),
            C2f(32, 32, n=1, shortcut=True),
        )
        self.stage3 = nn.Sequential(
            RepBlock(32, 64, 2),
            C2f(64, 64, n=2, shortcut=True),
        )
        self.stage4 = nn.Sequential(
            RepBlock(64, 128, 2),
            C2f(128, 128, n=2, shortcut=True),
        )
        self.stage5 = nn.Sequential(
            TransitionMSBlock(128, 256, stride=2),
            TransitionMSBlock(256, 256),
            SPPF(256, 256, k=5),
        )

        self.top4 = C2f(256 + 128, 128, n=1)
        self.top3 = C2f(128 + 64, 64, n=1)
        self.top2 = C2f(64 + 32, 32, n=1)

        self.down3 = RepBlock(32, 64, 2)
        self.down4 = RepBlock(192, 128, 2)
        self.bottom4 = C2f(128 + 128 + 128, 128, n=1)

        self.down5 = RepBlock(128, 256, 2)
        self.bottom5 = TransitionMSBlock(256 + 256, 256)

        self.fusions = nn.ModuleList([
            WeightedConcat(2),
            WeightedConcat(2),
            WeightedConcat(2),
            WeightedConcat(3),
            WeightedConcat(3),
            WeightedConcat(2),
        ])

        self.output_channels = [32, 192, 128, 256]
        self.strides = [4, 8, 16, 32]
        self.heads = nn.ModuleList([
            SlimRawHead(channels)
            for channels in self.output_channels
        ])

        self._initialize_native_biases()

    @torch.no_grad()
    def _initialize_native_biases(self):
        # Reuse installed YOLO initialization.
        # This is a reconstruction choice, not a paper-specified setting.
        from ultralytics.nn.modules.head import Detect

        # Constructing the reference head must not consume the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            reference = Detect(nc=1, ch=tuple(self.output_channels))
            reference.stride = torch.tensor(
                self.strides, dtype=torch.float32
            )
            reference.bias_init()

        for index, head in enumerate(self.heads):
            box_bias = reference.cv2[index][-1].bias
            cls_bias = reference.cv3[index][-1].bias

            if head.box[-1].bias.shape != box_bias.shape:
                raise RuntimeError("Native box bias shape mismatch.")
            if head.cls[-1].bias.shape != cls_bias.shape:
                raise RuntimeError("Native class bias shape mismatch.")

            head.box[-1].bias.copy_(box_bias)
            head.cls[-1].bias.copy_(cls_bias)

    @staticmethod
    def up(x, reference):
        return F.interpolate(
            x, size=reference.shape[-2:], mode="nearest"
        )

    def features(self, x):
        p2 = self.stage2(self.stem(x))
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        t4 = self.top4(self.fusions[0]([self.up(p5, p4), p4]))
        t3 = self.top3(self.fusions[1]([self.up(t4, p3), p3]))
        o2 = self.top2(self.fusions[2]([self.up(t3, p2), p2]))

        o3 = self.fusions[3]([self.down3(o2), t3, p3])
        o4 = self.bottom4(
            self.fusions[4]([self.down4(o3), t4, p4])
        )
        o5 = self.bottom5(
            self.fusions[5]([self.down5(o4), p5])
        )
        return [o2, o3, o4, o5]

    def forward(self, x):
        features = self.features(x)
        predictions = [
            head(feature)
            for head, feature in zip(self.heads, features)
        ]
        boxes = torch.cat([
            box.flatten(2) for box, score in predictions
        ], dim=2)
        scores = torch.cat([
            score.flatten(2) for box, score in predictions
        ], dim=2)

        return {
            "boxes": boxes,
            "scores": scores,
            "feats": features,
        }


def verify():
    torch.manual_seed(42)
    model = S2HeadReconstruction()

    print("RECONSTRUCTION ASSUMPTIONS")
    for key, value in ASSUMPTIONS.items():
        print(f"{key}: {value}")

    print("\nParameters:", sum(p.numel() for p in model.parameters()))
    print("Device: CPU float32")

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 640, 640))

    assert output["boxes"].shape == (1, 64, 34000)
    assert output["scores"].shape == (1, 1, 34000)
    assert torch.isfinite(output["boxes"]).all()
    assert torch.isfinite(output["scores"]).all()

    for stride, channels, feature in zip(
        model.strides, model.output_channels, output["feats"]
    ):
        assert feature.shape == (
            1, channels, 640 // stride, 640 // stride
        )
        print(
            f"PASS: stride={stride}, "
            f"feature={tuple(feature.shape)}"
        )

    del output

    # Smaller input checks the complete backward path cheaply.
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(torch.randn(2, 3, 128, 128))
    loss = (
        output["boxes"].square().mean()
        + output["scores"].square().mean()
    )
    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"No gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), name

    for index, fusion in enumerate(model.fusions):
        assert fusion.weights.grad.abs().sum() > 0, index

    print("PASS: complete backbone/neck/head backward path")
    print("Raw box shape at 640: (1, 64, 34000)")
    print("Raw class shape at 640: (1, 1, 34000)")
    print("\nS2HEAD ASSEMBLY CHECK PASS")
    print("Untrained reconstruction. No accuracy claim.")
    print("No dataset access or checkpoint changes.")


if __name__ == "__main__":
    verify()
