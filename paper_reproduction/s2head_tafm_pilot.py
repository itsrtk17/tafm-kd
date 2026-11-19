
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import ultralytics
from torch import nn
from ultralytics import YOLO

from s2head_native import (
    NativeS2HeadModel, NativeS2HeadTrainer,
    OneBatchFinished, start_check, batch_check,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from headlab_kd.losses import decode, aligned_giou, reliable_mask
RUNS = ROOT / "results/s2head_reconstruction_v1"
STUDENT = RUNS / "baseline_100ep_seed42/weights/best.pt"
TEACHER = RUNS / "teacher_wide_100ep_seed42/weights/best.pt"
GATE = TEACHER.parent.parent / "teacher_comparison.json"
REFERENCE = ROOT / "evaluations/s2head_vs_p2_val640_v1/summary.json"
DATA = ROOT / "configs/scut_dedup_v1/scut_head.yaml"
CONFIG = ROOT / "paper_reproduction/s2head_reconstruction.yaml"
OUTPUT = RUNS / "matched_tafm_pilot_v1"
STATE = {}


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def input_channels(block):
    return next(m.in_channels for m in block.modules() if isinstance(m, nn.Conv2d))


class TAFMStudent(NativeS2HeadModel):
    def loss(self, batch, preds=None):
        if not self.training:
            return super().loss(batch, preds)
        if preds is not None:
            raise RuntimeError("Expected the native training forward path.")

        student_head = self.model[-1]
        teacher = STATE["teacher"]
        teacher_head = teacher.model[-1]
        captured = {}
        handles = []
        try:
            for name, towers in (("box", student_head.cv2), ("cls", student_head.cv3)):
                for i, tower in enumerate(towers):
                    def capture(module, inputs, output, key=(name, i)):
                        captured[key] = output
                    handles.append(tower[0].register_forward_hook(capture))
            preds = self.forward(batch["img"])
        finally:
            for handle in handles:
                handle.remove()

        detection, items = super().loss(batch, preds)
        with torch.no_grad():
            features = teacher.model[0].features(batch["img"])
            target = teacher_head.forward_head(
                features, box_head=teacher_head.cv2, cls_head=teacher_head.cv3
            )
            target_boxes, centers, _ = decode(target, teacher_head)
            labels = {
                "bboxes": batch["bboxes"].to(target_boxes.device),
                "batch_idx": batch["batch_idx"].reshape(-1).to(target_boxes.device),
            }
            height, width = batch["img"].shape[-2:]
            if height != width:
                raise RuntimeError("This pilot expects square training inputs.")
            mask = reliable_mask(target_boxes, centers, labels, width, 0.25)
            target_probability = target["scores"].sigmoid()

        crossed = {"feats": features}
        for name, key, towers in (
            ("box", "boxes", teacher_head.cv2),
            ("cls", "scores", teacher_head.cv3),
        ):
            outputs = []
            for i, tower in enumerate(towers):
                feature = self.tafm_adapters[name][i](captured[name, i])
                # Teacher weights are frozen; keep gradients through this path.
                outputs.append(tower[2](tower[1](feature)).flatten(2))
            crossed[key] = torch.cat(outputs, dim=2)
            if crossed[key].shape != target[key].shape:
                raise RuntimeError(f"Crossed {key} grid differs from teacher grid.")

        cross_boxes, _, _ = decode(crossed, teacher_head)
        logits = crossed["scores"]
        cls_per_anchor = (
            F.binary_cross_entropy_with_logits(
                logits, target_probability, reduction="none"
            ) * (logits.sigmoid() - target_probability).abs()
        ).mean(dim=1)
        reg_per_anchor = (1 - aligned_giou(cross_boxes, target_boxes)).square()

        # Explicit reconstruction choice: mean over selected foreground positions.
        count = mask.sum()
        denominator = count.clamp_min(1)
        kd_cls = (cls_per_anchor * mask).sum() / denominator
        kd_reg = (reg_per_anchor * mask).sum() / denominator
        STATE["last"] = torch.stack((kd_cls.detach(), kd_reg.detach(), count.float()))
        STATE["epoch_stats"] = STATE["epoch_stats"] + STATE["last"]
        return detection.sum() + batch["img"].shape[0] * (kd_cls + kd_reg), items


class PilotTrainer(NativeS2HeadTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None or self.data["nc"] != 1:
            raise RuntimeError("Expected the specified one-class student checkpoint.")
        model = YOLO(str(STUDENT), task="detect").model.cpu().float()
        if type(model) is not NativeS2HeadModel:
            raise TypeError("Unexpected student model class.")
        if hasattr(model, "criterion"):
            del model.criterion
        model.requires_grad_(True)
        student_head = model.model[-1]
        student_head.dfl.requires_grad_(False)

        if STATE["kd"]:
            teacher_head = STATE["teacher"].model[-1]
            if (
                student_head.stride.tolist() != teacher_head.stride.tolist()
                or student_head.reg_max != teacher_head.reg_max
                or student_head.nc != teacher_head.nc
            ):
                raise RuntimeError("Incompatible teacher and student heads.")
            model.__class__ = TAFMStudent
            # Do not advance the data/augmentation RNG relative to the control.
            with torch.random.fork_rng(devices=[]):
                model.tafm_adapters = nn.ModuleDict({
                    name: nn.ModuleList(
                        nn.Conv2d(
                            input_channels(s[1]), input_channels(t[1]), 1,
                            bias=False,
                        )
                        for s, t in zip(stowers, ttowers)
                    )
                    for name, stowers, ttowers in (
                        ("box", student_head.cv2, teacher_head.cv2),
                        ("cls", student_head.cv3, teacher_head.cv3),
                    )
                })
        return model


def check_setup(trainer):
    if STATE["kd"]:
        teacher = STATE["teacher"].to(trainer.device)
        STATE["teacher_hash"] = fingerprint(teacher)
        ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert all(id(p) in ids for p in trainer.model.tafm_adapters.parameters())
        assert all(id(p) not in ids for p in teacher.parameters())
    if STATE["smoke"]:
        start_check(trainer)


def epoch_start(trainer):
    STATE["epoch_stats"] = torch.zeros(3, device=trainer.device)


def check_teacher():
    teacher = STATE["teacher"]
    assert not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert fingerprint(teacher) == STATE["teacher_hash"], "Teacher state changed."


def epoch_end(trainer):
    if STATE["kd"]:
        check_teacher()
        cls, reg, count = STATE["epoch_stats"].cpu().tolist()
        if not count or not torch.isfinite(STATE["epoch_stats"]).all().item():
            raise RuntimeError("Invalid KD losses or no selected teacher positions.")
        batches = len(trainer.train_loader)
        print(f"KD epoch mean: cls={cls/batches:.6f}, reg={reg/batches:.6f}, selected={int(count)}")


def smoke_end(trainer):
    cls, reg, count = STATE["last"].cpu().tolist()
    assert count > 0 and cls > 0 and reg > 0
    for name, adapters in trainer.model.tafm_adapters.items():
        grads = [p.grad for p in adapters.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads), name
        assert any(torch.count_nonzero(g).item() for g in grads), name
    check_teacher()
    print(f"TAFM-KD terms: cls={cls:.6f}, reg={reg:.6f}, selected={int(count)}")
    batch_check(trainer)


def train(mode):
    STATE.update(smoke=mode == "smoke", kd=mode != "control")
    output = OUTPUT / mode
    if output.exists():
        raise SystemExit(f"Run already exists: {output}. Nothing overwritten.")
    if STATE["kd"]:
        STATE["teacher"] = YOLO(str(TEACHER), task="detect").model.float().eval()
        STATE["teacher"].requires_grad_(False)

    output.mkdir(parents=True, exist_ok=False)
    trainer = PilotTrainer(overrides=dict(
        model=str(CONFIG), data=str(DATA), task="detect", mode="train",
        device="mps", imgsz=640, batch=1, nbs=8, workers=0,
        epochs=1 if STATE["smoke"] else 3,
        optimizer="SGD", lr0=0.0002, lrf=0.1, momentum=0.937,
        weight_decay=0.0005, cos_lr=True, warmup_epochs=0.0,
        mosaic=0.0, close_mosaic=0, amp=False, seed=42,
        deterministic=True, pretrained=False, patience=0,
        save=not STATE["smoke"], plots=False, val=not STATE["smoke"],
        conf=0.001, iou=0.7, max_det=1000,
        project=str(OUTPUT), name=mode, exist_ok=True,
    ))
    trainer.add_callback("on_train_start", check_setup)
    trainer.add_callback("on_train_epoch_start", epoch_start)
    trainer.add_callback("on_train_epoch_end", epoch_end)
    if STATE["smoke"]:
        trainer.add_callback("on_train_batch_end", smoke_end)
    try:
        trainer.train()
    except OneBatchFinished:
        if not STATE["smoke"]:
            raise
        print("TAFM-KD ONE-BATCH PASS")
        return
    if STATE["smoke"]:
        raise RuntimeError("Expected the one-batch stopping point.")


def compare():
    reference = json.loads(REFERENCE.read_text())
    checkpoints = {mode: OUTPUT / mode / "weights/best.pt" for mode in ("control", "kd")}
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise SystemExit(f"Missing checkpoint: {checkpoint}")

    report = OUTPUT / "matched_validation.json"
    if report.exists():
        raise SystemExit(f"Report already exists: {report}")
    results = {}
    for part in ("part_a", "part_b"):
        rows = {}
        for mode, checkpoint in checkpoints.items():
            model = YOLO(str(checkpoint), task="detect")
            metrics = model.val(
                data=reference["configs"][part], project=str(OUTPUT / "validation"),
                name=f"{part}_{mode}", **reference["settings"],
            )
            rows[mode] = float(metrics.box.map)
            del model, metrics
            gc.collect()
            torch.mps.empty_cache()
        original = reference["results"][part]["S2Head-reconstruction"]["map50_95"]
        results[part] = {
            "original": original, **rows,
            "kd_minus_control_pp": 100 * (rows["kd"] - rows["control"]),
            "kd_minus_original_pp": 100 * (rows["kd"] - original),
        }
    report.write_text(json.dumps(results, indent=2))
    print("\n========== MATCHED TAFM-KD VALIDATION AT 640 ==========")
    for part, row in results.items():
        print(
            f"{part}: original={100*row['original']:.2f}% | "
            f"control={100*row['control']:.2f}% | KD={100*row['kd']:.2f}% | "
            f"KD vs control={row['kd_minus_control_pp']:+.2f} pp | "
            f"KD vs original={row['kd_minus_original_pp']:+.2f} pp"
        )
    print("Report:", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "control", "kd", "compare"])
    mode = parser.parse_args().mode
    if ultralytics.__version__ != "8.4.176" or not torch.backends.mps.is_available():
        raise SystemExit("Expected Ultralytics 8.4.176 and MPS.")
    for path in (STUDENT, TEACHER, GATE, REFERENCE, DATA, CONFIG):
        if not path.is_file():
            raise SystemExit(f"Missing file: {path}")
    gate = json.loads(GATE.read_text())
    if (
        gate["teacher_gate_passed"] is not True
        or Path(gate["student"]).resolve() != STUDENT.resolve()
        or Path(gate["teacher"]).resolve() != TEACHER.resolve()
    ):
        raise SystemExit("Teacher gate does not match these checkpoints.")
    if mode == "compare":
        compare()
    else:
        train(mode)
