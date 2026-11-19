from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import cv2

BASE = Path(__file__).resolve().parents[1] / "data" / "raw"
PARTS = ["SCUT_HEAD_Part_A", "SCUT_HEAD_Part_B"]

for part in PARTS:
    root = BASE / part
    image_dir = root / "JPEGImages"
    annotation_dir = root / "Annotations"

    print(f"\n===== {part} =====")

    total_boxes = 0
    invalid_boxes = 0
    empty_annotations = 0
    malformed_xml = 0
    dimension_mismatches = 0
    unreadable_images = 0
    class_counts = Counter()
    examples = []

    for xml_path in sorted(annotation_dir.glob("*.xml")):
        image_path = image_dir / f"{xml_path.stem}.jpg"

        try:
            tree = ET.parse(xml_path)
            annotation = tree.getroot()
        except Exception as exc:
            malformed_xml += 1
            if len(examples) < 5:
                examples.append(f"Malformed XML: {xml_path.name}: {exc}")
            continue

        try:
            width = int(annotation.findtext("size/width"))
            height = int(annotation.findtext("size/height"))
        except (TypeError, ValueError):
            malformed_xml += 1
            if len(examples) < 5:
                examples.append(f"Missing/invalid dimensions: {xml_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            unreadable_images += 1
            if len(examples) < 5:
                examples.append(f"Unreadable/missing image: {image_path.name}")
        elif image.shape[1] != width or image.shape[0] != height:
            dimension_mismatches += 1
            if len(examples) < 5:
                examples.append(
                    f"Dimension mismatch: {xml_path.name}; "
                    f"XML={width}x{height}, image="
                    f"{image.shape[1]}x{image.shape[0]}"
                )

        objects = annotation.findall("object")
        if not objects:
            empty_annotations += 1

        for obj in objects:
            total_boxes += 1
            class_name = obj.findtext("name", default="UNKNOWN").strip()
            class_counts[class_name] += 1

            try:
                box = obj.find("bndbox")
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                valid = (
                    xmin >= 0 and ymin >= 0
                    and xmax > xmin and ymax > ymin
                    and xmax <= width and ymax <= height
                )
            except (AttributeError, TypeError, ValueError):
                valid = False

            if not valid:
                invalid_boxes += 1
                if len(examples) < 5:
                    examples.append(f"Invalid box in {xml_path.name}")

    print("XML files checked:", len(list(annotation_dir.glob("*.xml"))))
    print("Total annotated objects:", total_boxes)
    print("Class counts:", dict(class_counts))
    print("Invalid bounding boxes:", invalid_boxes)
    print("Empty annotations:", empty_annotations)
    print("Malformed XML/dimensions:", malformed_xml)
    print("Image dimension mismatches:", dimension_mismatches)
    print("Unreadable/missing images:", unreadable_images)

    if examples:
        print("Examples requiring inspection:")
        for example in examples:
            print(" -", example)
