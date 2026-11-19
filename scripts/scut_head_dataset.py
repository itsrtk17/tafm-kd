
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image


class SCUTHeadDataset(Dataset):
    """SCUT-HEAD dataset using the official split files."""

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        split_file = (
            self.root / "ImageSets" / "Main" / f"{split}.txt"
        )
        with split_file.open("r", encoding="utf-8") as file:
            self.image_ids = [
                line.strip()
                for line in file
                if line.strip()
            ]

        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = self.image_dir / f"{image_id}.jpg"
        annotation_path = (
            self.annotation_dir / f"{image_id}.xml"
        )

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        annotation = ET.parse(annotation_path).getroot()
        boxes = []

        for obj in annotation.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue

            try:
                xmin, ymin, xmax, ymax = [
                    float(box.findtext(key))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                ]
            except (TypeError, ValueError):
                continue

            # Reject degenerate or out-of-image boxes.
            # Use actual image dimensions, not XML dimensions.
            if (
                xmin < 0 or ymin < 0
                or xmax <= xmin or ymax <= ymin
                or xmax > width or ymax > height
            ):
                continue

            boxes.append([xmin, ymin, xmax, ymax])

        target = {
            "boxes": torch.tensor(
                boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.ones(
                len(boxes), dtype=torch.int64
            ),
            "image_id": torch.tensor([index]),
            "orig_size": torch.tensor([height, width]),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target
