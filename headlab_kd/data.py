from pathlib import Path
import hashlib
import random
import numpy as np
import yaml
from PIL import Image


class HeadDataset:
    """Single-class YOLO labels, square letterbox, optional horizontal flip.

    Deliberately no mosaic or corruption augmentation in the matched continuation.
    All image/label paths come from the user's existing split lists.
    """

    def __init__(self, config, size=640, seed=42):
        self.config = Path(config).resolve()
        cfg = yaml.safe_load(self.config.read_text())
        self.root = Path(cfg['path']).expanduser().resolve()
        self.listing = Path(cfg['train']).expanduser()
        if not self.listing.is_absolute():
            self.listing = self.root / self.listing
        if not self.listing.is_file():
            raise ValueError('This runner requires an existing train image-list file.')
        self.images = []
        for line in self.listing.read_text().splitlines():
            if line.strip():
                path = Path(line.strip()).expanduser()
                if not path.is_absolute():
                    path = self.listing.parent / path
                self.images.append(path.resolve())
        if not self.images or len(self.images) != len(set(self.images)):
            raise ValueError('Empty train list or repeated image paths')
        self.size, self.seed, self.epoch = size, seed, 0
        fingerprint = hashlib.sha256(self.listing.read_bytes())
        fingerprint.update(self.config.read_bytes())
        for split in ['val', 'test']:
            entry = Path(cfg[split]).expanduser()
            if not entry.is_absolute():
                entry = self.root / entry
            if not entry.is_file():
                raise ValueError(f'Expected an existing {split} image list: {entry}')
            fingerprint.update(entry.read_bytes())
        self.labels = []
        for image in self.images:
            relative = image.relative_to(self.root / 'images')
            label = (self.root / 'labels' / relative).with_suffix('.txt')
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f'Missing image/label: {image}, {label}')
            content = label.read_bytes()
            fingerprint.update(str(label).encode())
            fingerprint.update(content)
            rows = []
            for line in content.decode().splitlines():
                if line.strip():
                    values = list(map(float, line.split()))
                    if len(values) != 5 or values[0] != 0:
                        raise ValueError(f'Expected one-class YOLO head labels: {label}')
                    box = np.array(values[1:], dtype=np.float32)
                    if not np.isfinite(box).all() or (box < 0).any() or (box > 1).any() or (box[2:] <= 0).any():
                        raise ValueError(f'Invalid normalized box in {label}')
                    if (box[:2]-box[2:]/2 < -1e-5).any() or (box[:2]+box[2:]/2 > 1+1e-5).any():
                        raise ValueError(f'Box exceeds image in {label}')
                    rows.append(box)
            self.labels.append(np.array(rows, dtype=np.float32).reshape(-1, 4))
        self.fingerprint = fingerprint.hexdigest()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        boxes = self.labels[index].copy()
        with Image.open(path) as image:
            # Do not apply EXIF rotation: labels refer to the stored image coordinates.
            image = image.convert('RGB')
            width, height = image.size
            ratio = min(self.size / width, self.size / height)
            new_w = max(1, round(width * ratio))
            new_h = max(1, round(height * ratio))
            left, top = (self.size-new_w)//2, (self.size-new_h)//2
            canvas = Image.new('RGB', (self.size, self.size), (114, 114, 114))
            canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
            if len(boxes):
                boxes[:, 0] = (boxes[:, 0] * new_w + left) / self.size
                boxes[:, 1] = (boxes[:, 1] * new_h + top) / self.size
                boxes[:, 2] *= new_w / self.size
                boxes[:, 3] *= new_h / self.size
            # Local RNG: adapters/teacher initialization cannot change augmentation choices.
            rng = random.Random(self.seed + 1000003*self.epoch + index)
            if rng.random() < .5:
                canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if len(boxes):
                    boxes[:, 0] = 1 - boxes[:, 0]
            pixels = np.asarray(canvas, dtype=np.float32).copy() / 255
        return pixels.transpose(2, 0, 1), boxes, str(path)
