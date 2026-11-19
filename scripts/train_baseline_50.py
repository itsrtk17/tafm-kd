
from ultralytics import YOLO
import torch

DATA = "configs/scut_dedup_v1/scut_head.yaml"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    model = YOLO("yolo11n.yaml")

    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=4,
        device=DEVICE,
        workers=0,
        seed=42,
        deterministic=True,
        patience=15,
        project="results/baseline",
        name="yolo11n_scut_head_50ep",
        exist_ok=True,
        plots=True,
        save=True,
    )

if __name__ == "__main__":
    main()
