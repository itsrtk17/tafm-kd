from pathlib import Path
from collections import Counter
import os
import gc
import numpy as np
import torch
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
data = root / "data/processed/scut_head"

models = {
    "YOLO11n Baseline":
        root / "runs/detect/results/baseline/"
        "yolo11n_scut_head_50ep/weights/best.pt",
    "YOLO11n-P2":
        root / "results/experiments/"
        "yolo11n_p2_50ep/weights/best.pt",
}

parts = {
    "Part A": "part_a_val_images.txt",
    "Part B": "part_b_val_images.txt",
}

device = (
    "mps"
    if os.getenv("FORCE_CPU") != "1"
    and torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)
print("Inference batch size: 1")


def calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(x2 - x1, 0)
        * np.maximum(y2 - y1, 0)
    )

    area1 = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

    area2 = (
        np.maximum(boxes[:, 2] - boxes[:, 0], 0)
        * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    )

    return intersection / np.maximum(
        area1 + area2 - intersection, 1e-9
    )


for model_name, weights in models.items():
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    model = YOLO(str(weights))

    print(f"\n========== {model_name} ==========")

    for part_name, list_name in parts.items():
        image_list = root / "configs" / list_name

        if not image_list.is_file():
            raise SystemExit(f"Missing image list: {image_list}")

        total = Counter()
        detected = Counter()
        processed = 0

        predictions = model.predict(
            source=str(image_list),
            imgsz=640,
            batch=1,
            conf=0.25,
            iou=0.7,
            max_det=300,
            device=device,
            rect=False,
            stream=True,
            save=False,
            verbose=False,
        )

        for result in predictions:
            processed += 1

            image = Path(result.path)
            h, w = result.orig_shape
            scale = min(640 / w, 640 / h)

            label = (
                data / "labels/val" / f"{image.stem}.txt"
            )

            if not label.is_file():
                raise SystemExit(f"Missing label: {label}")

            gt_boxes = []
            sizes = []

            for line in label.read_text().splitlines():
                cls, cx, cy, bw, bh = map(
                    float, line.split()
                )

                gt_boxes.append([
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ])

                area = (bw * w * scale) * (bh * h * scale)

                size = (
                    "Small" if area < 32**2
                    else "Medium" if area < 96**2
                    else "Large"
                )

                sizes.append(size)
                total[size] += 1

            if not gt_boxes:
                continue

            pred_boxes = result.boxes.xyxy.cpu().numpy()

            if len(pred_boxes) == 0:
                continue

            scores = result.boxes.conf.cpu().numpy()
            gt_boxes = np.asarray(gt_boxes)

            matched = set()

            for idx in np.argsort(-scores):
                ious = calculate_iou(
                    pred_boxes[idx], gt_boxes
                )

                if matched:
                    ious[list(matched)] = -1

                best = int(np.argmax(ious))

                if ious[best] >= 0.5:
                    matched.add(best)
                    detected[sizes[best]] += 1

            if processed % 100 == 0:
                print(
                    f"{part_name}: {processed} images processed"
                )

        expected = 400 if part_name == "Part A" else 462
        if processed != expected:
            raise RuntimeError(
                f"Expected {expected} images, got {processed}"
            )

        print(f"\n{part_name} ({processed} images):")

        for size in ["Small", "Medium", "Large"]:
            n = total[size]

            if n:
                recall = detected[size] / n * 100
                print(
                    f"{size}: {detected[size]}/{n} "
                    f"({recall:.2f}% recall)"
                )
            else:
                print(f"{size}: No annotated heads")

    del model
    gc.collect()

    if device == "mps":
        torch.mps.empty_cache()

print("\nHead-size recall analysis completed!")
