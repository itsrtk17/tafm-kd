
from pathlib import Path
from scut_head_dataset import SCUTHeadDataset

project = Path(__file__).resolve().parents[1]
data_dir = project / "data" / "raw"

for part in ("SCUT_HEAD_Part_A", "SCUT_HEAD_Part_B"):
    for split in ("train", "val", "test"):
        dataset = SCUTHeadDataset(
            root=data_dir / part,
            split=split,
        )

        image, target = dataset[0]

        print(
            part,
            split,
            "images:", len(dataset),
            "image size:", image.size,
            "boxes:", target["boxes"].shape,
        )
