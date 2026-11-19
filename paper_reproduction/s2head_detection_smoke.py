
"""Real-annotation detection-loss check. No training run."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from ultralytics.data.utils import img2label_paths
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from s2head_model import S2HeadReconstruction


class LossBridge(nn.Module):
    """Supply the metadata expected by native v8DetectionLoss."""

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.register_buffer(
            "stride", torch.tensor(network.strides, dtype=torch.float32)
        )
        # Native loss gains; not claimed as paper-specified values.
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    @property
    def model(self):
        return [
            SimpleNamespace(
                nc=1, reg_max=16, stride=self.stride
            )
        ]


def first_training_sample(root):
    config = root / "configs/scut_dedup_v1/scut_head.yaml"
    data = yaml.safe_load(config.read_text())
    base = Path(data["path"]).expanduser()
    if not base.is_absolute():
        base = (root / base).resolve()

    entries = data["train"]
    if isinstance(entries, str):
        entries = [entries]

    images = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for entry in entries:
        source = Path(entry).expanduser()
        if not source.is_absolute():
            source = base / source

        if source.is_dir():
            images.extend(sorted(
                p for p in source.rglob("*")
                if p.suffix.lower() in extensions and p.is_file()
            ))
        elif source.suffix.lower() == ".txt":
            for line in source.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                image = Path(line).expanduser()
                if not image.is_absolute():
                    image = source.parent / image
                images.append(image.resolve())
        else:
            raise RuntimeError(f"Unsupported training entry: {source}")

    for image in images:
        label = Path(img2label_paths([str(image)])[0])
        if image.is_file() and label.is_file():
            lines = [
                line.split()
                for line in label.read_text().splitlines()
                if line.strip()
            ]
            if lines:
                if any(len(row) != 5 for row in lines):
                    raise RuntimeError(f"Invalid detection label: {label}")
                return image, label, np.asarray(lines, dtype=np.float32)

    raise RuntimeError("No annotated training image found.")


def prepare(image_path, labels, size=640):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")

    height, width = image.shape[:2]
    assert np.isfinite(labels).all()
    assert (labels[:, 0] == 0).all(), "Expected one head class."
    assert ((labels[:, 1:] >= 0) & (labels[:, 1:] <= 1)).all()
    assert (labels[:, 3:] > 0).all()

    # Fixed-size centered letterbox; transform labels consistently.
    ratio = min(size / height, size / width)
    resized_w = round(width * ratio)
    resized_h = round(height * ratio)
    left = round((size - resized_w) / 2 - 0.1)
    top = round((size - resized_h) / 2 - 0.1)

    resized = cv2.resize(
        image, (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = resized

    boxes = labels[:, 1:5].copy()
    boxes[:, [0, 2]] *= width * ratio
    boxes[:, [1, 3]] *= height * ratio
    boxes[:, 0] += left
    boxes[:, 1] += top
    boxes /= size

    # BGR -> RGB, BCHW, float32 in [0, 1].
    pixels = np.ascontiguousarray(
        canvas[:, :, ::-1].transpose(2, 0, 1)
    )
    tensor = torch.from_numpy(pixels).float().unsqueeze(0) / 255

    return {
        "img": tensor,
        "batch_idx": torch.zeros(len(labels), dtype=torch.long),
        "cls": torch.from_numpy(labels[:, :1].copy()),
        "bboxes": torch.from_numpy(boxes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable.")

    torch.manual_seed(42)
    image, label, annotations = first_training_sample(
        Path(args.root).expanduser().resolve()
    )
    batch = {
        key: value.to(args.device)
        for key, value in prepare(image, annotations).items()
    }

    network = S2HeadReconstruction().to(args.device).train()
    bridge = LossBridge(network).to(args.device)
    criterion = v8DetectionLoss(bridge)

    predictions = network(batch["img"])
    assigned, components, items = (
        criterion.get_assigned_targets_and_loss(predictions, batch)
    )
    loss = components.sum() * batch["img"].shape[0]

    assert torch.isfinite(loss).item(), "Non-finite detection loss."
    foreground = int(assigned[0].sum().item())
    assert foreground > 0, "No foreground assignments; inspect setup."

    loss.backward()

    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"No gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), name

    stem_gradient = network.stem[0][0].weight.grad.abs().sum().item()
    assert stem_gradient > 0

    # Decode native DFL distributions into pixel-space xyxy boxes.
    with torch.no_grad():
        anchors, strides = make_anchors(
            predictions["feats"], bridge.stride, 0.5
        )
        decoded = criterion.bbox_decode(
            anchors,
            predictions["boxes"].permute(0, 2, 1).contiguous(),
        ) * strides

        assert decoded.shape == (1, 34000, 4)
        assert torch.isfinite(decoded).all()
        assert (decoded[..., 2:] >= decoded[..., :2]).all()

    print("\n========== REAL DETECTION LOSS CHECK ==========")
    print("Device:", args.device)
    print("Training image:", image)
    print("Label file:", label)
    print("Annotated heads:", len(annotations))
    print("Assigned foreground positions:", foreground)
    print("Loss gains: box=7.5, cls=0.5, dfl=1.5")
    print("Box / classification / DFL:", components.detach().cpu().tolist())
    print("Total loss:", loss.item())
    print("Stem gradient magnitude:", stem_gradient)
    print("Decoded xyxy shape:", tuple(decoded.shape))
    print("All trainable parameters: finite gradients")
    print("\nREAL DETECTION LOSS CHECK PASS")
    print("Random initialization; no accuracy measurement.")
    print("No optimizer step, checkpoint writes or dataset changes.")


if __name__ == "__main__":
    main()
