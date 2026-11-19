
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import cv2

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
OUTPUT = PROJECT / "data" / "processed" / "scut_head"

PARTS = ["SCUT_HEAD_Part_A", "SCUT_HEAD_Part_B"]
SPLITS = ["train", "val", "test"]


def convert_box(obj, width, height):
    """Convert a valid Pascal VOC box to normalized YOLO format."""
    box = obj.find("bndbox")
    if box is None:
        return None

    try:
        xmin, ymin, xmax, ymax = [
            float(box.findtext(key))
            for key in ["xmin", "ymin", "xmax", "ymax"]
        ]
    except (TypeError, ValueError):
        return None

    # Clamp only small coordinate overruns to image boundaries.
    # Reject degenerate or completely out-of-image boxes.
    if xmax <= xmin or ymax <= ymin:
        return None

    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return (
        f"0 {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def main():
    for split in SPLITS:
        (OUTPUT / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (OUTPUT / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )

    total_images = 0
    total_boxes = 0
    skipped_boxes = 0

    for part in PARTS:
        root = RAW / part
        split_dir = root / "ImageSets" / "Main"

        for split in SPLITS:
            ids = (
                split_dir / f"{split}.txt"
            ).read_text(encoding="utf-8").split()

            for image_id in ids:
                image_path = root / "JPEGImages" / f"{image_id}.jpg"
                xml_path = root / "Annotations" / f"{image_id}.xml"

                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(
                        f"Could not read image: {image_path}"
                    )

                height, width = image.shape[:2]
                annotation = ET.parse(xml_path).getroot()

                label_lines = []
                for obj in annotation.findall("object"):
                    line = convert_box(obj, width, height)
                    if line is None:
                        skipped_boxes += 1
                    else:
                        label_lines.append(line)

                # Prefix part name to keep image IDs unique.
                output_id = f"{part}_{image_id}"
                output_image = (
                    OUTPUT / "images" / split /
                    f"{output_id}.jpg"
                )
                output_label = (
                    OUTPUT / "labels" / split /
                    f"{output_id}.txt"
                )

                # Copy each image only once per assigned split.
                shutil.copy2(image_path, output_image)
                output_label.write_text(
                    "\n".join(label_lines) +
                    ("\n" if label_lines else ""),
                    encoding="utf-8",
                )

                total_images += 1
                total_boxes += len(label_lines)

    print("Dataset conversion completed.")
    print("Output:", OUTPUT)
    print("Images:", total_images)
    print("Valid boxes written:", total_boxes)
    print("Boxes skipped:", skipped_boxes)


if __name__ == "__main__":
    main()
