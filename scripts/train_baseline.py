
from ultralytics import YOLO
import torch

DATA = "configs/scut_dedup_v1/scut_head.yaml"
MODEL = "yolo11n.yaml"

# Use Apple's GPU when supported.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    print("Device:", DEVICE)
    print("Dataset:", DATA)
    print("Model:", MODEL)

    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=10,
        imgsz=640,
        batch=4,
        device=DEVICE,
        workers=0,
        seed=42,
        deterministic=True,
        patience=5,
        project="results/baseline",
        name="yolo11n_scut_head",
        exist_ok=True,
        plots=True,
        save=True,
    )

    print("\nBaseline training finished.")
    print("Best weights: results/baseline/yolo11n_scut_head/weights/best.pt")

if __name__ == "__main__":
    main()
