
"""Wider/deeper S2Head reconstruction for the TAFM-KD teacher."""

from pathlib import Path

import torch
import ultralytics
from torch import nn
from ultralytics.nn.modules import C2f, SPPF
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import initialize_weights

from s2head_blocks import RepBlock
from s2head_fusion import WeightedConcat
from s2head_model import (
    S2HeadReconstruction, SlimRawHead, TransitionMSBlock, cbs,
)
from s2head_native import (
    NativeS2HeadModel, NativeS2HeadTrainer,
    OneBatchFinished, start_check, batch_check,
)


class WideS2HeadFeatures(S2HeadReconstruction):
    """Reuse the student's feature connections with wider/deeper layers."""

    def __init__(self):
        nn.Module.__init__(self)

        # Explicit reconstruction choices, not author-supplied widths.
        stem, p2, p3, p4, p5 = 24, 40, 80, 160, 320
        self.stem = cbs(3, stem, 3, 2)
        self.stage2 = nn.Sequential(
            RepBlock(stem, p2, 2), C2f(p2, p2, n=2, shortcut=True)
        )
        self.stage3 = nn.Sequential(
            RepBlock(p2, p3, 2), C2f(p3, p3, n=3, shortcut=True)
        )
        self.stage4 = nn.Sequential(
            RepBlock(p3, p4, 2), C2f(p4, p4, n=3, shortcut=True)
        )
        self.stage5 = nn.Sequential(
            TransitionMSBlock(p4, p5, stride=2),
            TransitionMSBlock(p5, p5),
            SPPF(p5, p5, k=5),
        )

        self.top4 = C2f(p5 + p4, p4, n=2)
        self.top3 = C2f(p4 + p3, p3, n=2)
        self.top2 = C2f(p3 + p2, p2, n=2)
        self.down3 = RepBlock(p2, p3, 2)
        self.down4 = RepBlock(3 * p3, p4, 2)
        self.bottom4 = C2f(3 * p4, p4, n=2)
        self.down5 = RepBlock(p4, p5, 2)
        self.bottom5 = TransitionMSBlock(2 * p5, p5)
        self.fusions = nn.ModuleList(
            WeightedConcat(n) for n in (2, 2, 2, 3, 3, 2)
        )
        self.output_channels = [p2, 3 * p3, p4, p5]
        self.strides = [4, 8, 16, 32]


def build_teacher():
    model = NativeS2HeadModel()
    student_params = sum(p.numel() for p in model.parameters())

    features = WideS2HeadFeatures()
    branches = [SlimRawHead(c) for c in features.output_channels]
    head = Detect(nc=1, ch=tuple(features.output_channels))
    head.cv2 = nn.ModuleList(branch.box for branch in branches)
    head.cv3 = nn.ModuleList(branch.cls for branch in branches)
    head.stride = torch.tensor(features.strides, dtype=torch.float32)
    head.bias_init()

    for i, module in enumerate((features, head)):
        module.i = i
        module.f = -1
        module.type = type(module).__name__
        module.np = sum(p.numel() for p in module.parameters())

    model.model = nn.ModuleList([features, head])
    model.stride = head.stride
    model.yaml = {
        "nc": 1,
        "channels": 3,
        "architecture": "s2head_wide_deep_teacher_reconstruction",
        "backbone_widths": [24, 40, 80, 160, 320],
        "backbone_c2f_repeats": [2, 3, 3],
        "neck_c2f_repeats": 2,
    }
    initialize_weights(model)

    teacher_params = sum(p.numel() for p in model.parameters())
    assert teacher_params > student_params
    print(f"Student parameters: {student_params:,}")
    print(f"Teacher parameters: {teacher_params:,}")
    print("Teacher head input channels:", features.output_channels)
    return model


class WideTeacherTrainer(NativeS2HeadTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None or self.data["nc"] != 1:
            raise RuntimeError("Expected fresh initialization and one class.")
        return build_teacher()


def smoke():
    if ultralytics.__version__ != "8.4.176":
        raise SystemExit("Expected Ultralytics 8.4.176.")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable.")

    root = Path(__file__).resolve().parents[1]
    config = root / "paper_reproduction/s2head_reconstruction.yaml"
    data = root / "configs/scut_dedup_v1/scut_head.yaml"
    for source in (config, data):
        if not source.is_file():
            raise SystemExit(f"Missing file: {source}")

    output = root / "results/s2head_reconstruction_v1/wide_teacher_smoke_v1"
    if output.exists():
        raise SystemExit(f"Run already exists: {output}. Nothing overwritten.")
    output.mkdir(parents=True, exist_ok=False)

    trainer = WideTeacherTrainer(overrides=dict(
        model=str(config), data=str(data), task="detect", mode="train",
        device="mps", imgsz=640, batch=1, workers=0, epochs=1,
        optimizer="SGD", lr0=0.01, momentum=0.937,
        weight_decay=0.0005, nbs=64, warmup_epochs=0.0,
        mosaic=1.0, close_mosaic=0, amp=False, seed=42,
        deterministic=True, pretrained=False, patience=0,
        save=False, plots=False, val=False, max_det=1000,
        project=str(output.parent), name=output.name, exist_ok=True,
    ))
    trainer.add_callback("on_train_start", start_check)
    trainer.add_callback("on_train_batch_end", batch_check)
    try:
        trainer.train()
    except OneBatchFinished:
        print("WIDE TAFM-KD TEACHER SMOKE PASS")
        return
    raise RuntimeError("Expected the one-batch stopping point.")
