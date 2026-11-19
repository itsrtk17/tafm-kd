
"""Native trainer integration for the documented S2Head reconstruction."""

from pathlib import Path

import torch
import ultralytics
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from s2head_model import S2HeadReconstruction, ASSUMPTIONS


class NativeS2HeadModel(DetectionModel):
    def __init__(self):
        # Build our architecture directly; do not parse a YOLO architecture.
        nn.Module.__init__(self)

        architecture = S2HeadReconstruction()
        branches = architecture.heads
        del architecture.heads

        # Reuse native decoding, DFL and prediction interfaces.
        with torch.random.fork_rng(devices=[]):
            detect = Detect(
                nc=1, ch=tuple(architecture.output_channels)
            )

        detect.cv2 = nn.ModuleList([head.box for head in branches])
        detect.cv3 = nn.ModuleList([head.cls for head in branches])
        detect.stride = torch.tensor(
            architecture.strides, dtype=torch.float32
        )

        for index, module in enumerate([architecture, detect]):
            module.i = index
            module.f = -1
            module.type = type(module).__name__
            module.np = sum(p.numel() for p in module.parameters())

        self.model = nn.ModuleList([architecture, detect])
        self.save = []
        self.stride = detect.stride
        self.names = {0: "head"}
        self.task = "detect"
        self.inplace = True
        self.args = get_cfg()
        self.yaml = {
            "nc": 1,
            "channels": 3,
            "architecture": "documented_s2head_reconstruction",
            "assumptions": ASSUMPTIONS,
        }

        # Apply native BN settings and activation conventions.
        initialize_weights(self)

    def _predict_once(
        self, x, profile=False, visualize=False, embed=None
    ):
        if embed is not None:
            raise NotImplementedError(
                "Embedding extraction is not implemented."
            )
        features = self.model[0].features(x)
        return self.model[1](features)


class NativeS2HeadTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data["nc"] != 1:
            raise RuntimeError("Expected one head class.")

        model = NativeS2HeadModel()
        if weights is not None:
            raise RuntimeError(
                "This first integration check requires fresh initialization."
            )
        return model


class OneBatchFinished(Exception):
    pass


def start_check(trainer):
    if trainer.ema is None:
        raise RuntimeError("Native EMA was not created.")

    head = trainer.model.model[-1]
    assert isinstance(head, Detect)
    assert head.nc == 1
    assert head.stride.tolist() == [4, 8, 16, 32]

    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    required_ids = {
        id(parameter)
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    }
    assert required_ids <= optimizer_ids

    trainer.headlab_initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    print(
        "\nNATIVE SETUP PASS: custom architecture; native head, "
        "optimizer, detection loss and EMA.",
        flush=True,
    )


def batch_check(trainer):
    assert torch.isfinite(trainer.loss).all().item()

    gradient_total = 0.0
    for name, parameter in trainer.model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            gradient_total += parameter.grad.detach().abs().sum().item()

        assert torch.equal(
            parameter.detach().cpu(),
            trainer.headlab_initial_parameters[name],
        ), f"Unexpected parameter update: {name}"

    assert gradient_total > 0, "Missing gradients."
    assert not trainer.optimizer.state, "Optimizer state changed."

    print("\n========== NATIVE S2HEAD ONE-BATCH PASS ==========")
    print("Native total loss:", trainer.loss.detach().sum().item())
    print("Gradient magnitude:", gradient_total)
    print("Accumulation:", trainer.accumulate)
    print("EMA: enabled")
    print("Optimizer membership: PASS")
    print("Parameters unchanged: PASS")
    print("Output folder:", trainer.save_dir)
    print(
        "Stopping after one batch; no optimizer step "
        "or full training performed.",
        flush=True,
    )
    raise OneBatchFinished()


def smoke():
    if ultralytics.__version__ != "8.4.176":
        raise RuntimeError(
            "This integration targets Ultralytics 8.4.176."
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable.")

    root = Path(__file__).resolve().parents[1]
    config = root / "paper_reproduction/s2head_reconstruction.yaml"
    data = root / "configs/scut_dedup_v1/scut_head.yaml"
    output = root / "results/s2head_reconstruction_v1/native_smoke"

    if not data.is_file():
        raise RuntimeError(f"Missing dataset configuration: {data}")
    if output.exists():
        raise RuntimeError(
            f"Output already exists: {output}. "
            "Send its output before rerunning; nothing overwritten."
        )

    output.mkdir(parents=True)

    trainer = NativeS2HeadTrainer(overrides={
        "model": str(config),
        "data": str(data),
        "task": "detect",
        "mode": "train",
        "device": "mps",
        "imgsz": 640,
        "batch": 1,
        "workers": 0,
        "epochs": 1,
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "nbs": 64,
        "warmup_epochs": 0.0,
        "amp": False,
        "seed": 42,
        "deterministic": True,
        "mosaic": 1.0,
        "close_mosaic": 0,
        "patience": 0,
        "save": False,
        "plots": False,
        "val": False,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
    })

    trainer.add_callback("on_train_start", start_check)
    trainer.add_callback("on_train_batch_end", batch_check)

    try:
        trainer.train()
    except OneBatchFinished:
        return
    raise RuntimeError("Expected one-batch stopping point was not reached.")
