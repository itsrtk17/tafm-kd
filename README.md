# TAFM-KD: Crowded Head Detection

Research implementation for **Improving Crowded Head Detection With Task-Aware Focal Modulation and Feature-Crossing Knowledge Distillation**.

This repository contains YOLO11-based crowded-head detection experiments, an S2Head-style architecture reconstruction, and experimental feature-crossing knowledge-distillation components.

> **Research scope:** This is an adapted research implementation, not an official implementation from the paper's authors and not a claim of exact reproduction of every component or result in the paper. See the implementation notes below.

## 1. Project overview

The project investigates crowded-head detection using the SCUT-HEAD dataset.

The repository includes:

- YOLO11 baseline training scripts.
- A P2 detection-head configuration for small-object detection.
- SCUT-HEAD dataset preparation and validation utilities.
- S2Head-style architecture and feature-module implementations.
- Experimental feature-crossing knowledge distillation (KD).
- Smoke tests and experiment comparison utilities.

## 2. Repository structure

```text
tafm-kd/
├── configs/
│   ├── scut_dedup_v1/
│   │   ├── scut_head.yaml
│   │   ├── part_a_val.yaml
│   │   ├── part_b_val.yaml
│   │   └── *_images.txt
│   ├── yolo11n_p2.yaml
│   └── yolo11s_p2_teacher.yaml
├── headlab_kd/
│   ├── __init__.py
│   ├── __main__.py
│   ├── data.py
│   └── losses.py
├── paper_reproduction/
│   ├── s2head_blocks.py
│   ├── s2head_detection_smoke.py
│   ├── s2head_fusion.py
│   ├── s2head_model.py
│   ├── s2head_msblock.py
│   ├── s2head_native.py
│   ├── s2head_reconstruction.yaml
│   ├── s2head_tafm_pilot.py
│   └── s2head_wide_teacher.py
├── scripts/
│   ├── compare_head_recall.py
│   ├── prepare_yolo_dataset.py
│   ├── scut_head_dataset.py
│   ├── test_dataset.py
│   ├── train_baseline.py
│   ├── train_baseline_50.py
│   └── validate_scut_head.py
├── .gitignore
├── README.md
└── requirements.txt
```

Dataset files, model checkpoints, generated training runs, and local experiment outputs are intentionally excluded from Git.

## 3. Requirements

- Python 3.11 recommended.
- PyTorch.
- Ultralytics.
- NumPy.
- OpenCV.
- PyYAML.
- Pillow.

The development environment used for the reported experiments was macOS on Apple Silicon with MPS acceleration, PyTorch 2.14.1, and Ultralytics 8.4.176.

Other operating systems and CUDA configurations have not been fully verified.

## 4. Installation

Clone the repository:

```bash
git clone https://github.com/itsrtk17/tafm-kd.git
cd tafm-kd
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the installation:

```bash
python -c "import torch, ultralytics, cv2; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

## 5. Dataset preparation

This project uses the SCUT-HEAD dataset, including Part A and Part B.

The expected raw dataset layout for the conversion and validation utilities is:

```text
data/
└── raw/
    ├── SCUT_HEAD_Part_A/
    │   ├── Annotations/
    │   ├── JPEGImages/
    │   └── ImageSets/
    │       └── Main/
    │           ├── train.txt
    │           ├── val.txt
    │           └── test.txt
    └── SCUT_HEAD_Part_B/
        ├── Annotations/
        ├── JPEGImages/
        └── ImageSets/
            └── Main/
                ├── train.txt
                ├── val.txt
                └── test.txt
```

Obtain the dataset from its authorized source and follow its applicable terms of use. Dataset images and annotations are not distributed in this repository.

### Convert raw annotations

```bash
python scripts/prepare_yolo_dataset.py
```

The conversion utility writes YOLO-format images and labels under:

```text
data/processed/scut_head/
```

**Important split note:** The conversion script uses the raw dataset's official `ImageSets/Main` split files. The reported clean experiments use the separately supplied deduplicated split manifests in `configs/scut_dedup_v1/`. Do not assume that running the converter recreates those exact deduplicated memberships. Check the image lists and split assignments before using converted data for experiment reproduction.

### Validate the raw dataset

```bash
python scripts/validate_scut_head.py
```

### Inspect dataset samples

```bash
python scripts/test_dataset.py
```

These utilities require the raw dataset to be installed in the expected location.

## 6. Dataset configuration

The clean experiments use:

```text
configs/scut_dedup_v1/scut_head.yaml
```

The dataset configuration defines one detection class:

```yaml
nc: 1
names:
  0: head
```

The split manifests reference image files under the processed dataset directory. The dataset itself is not included in this repository.

Run training and validation commands from the repository root so relative dataset paths resolve correctly.

## 7. Baseline training

Train the 10-epoch baseline:

```bash
python scripts/train_baseline.py
```

Train the 50-epoch baseline:

```bash
python scripts/train_baseline_50.py
```

Both scripts use the deduplicated dataset configuration. They select Apple MPS when available and otherwise use CPU.

Training creates local output files under `results/`.

## 8. P2 model configuration

The repository includes:

- `configs/yolo11n_p2.yaml` — YOLO11n P2 student configuration.
- `configs/yolo11s_p2_teacher.yaml` — YOLO11s P2 teacher configuration.

The P2 detection level adds a higher-resolution feature level intended to help detect small heads in crowded scenes.

## 9. S2Head-style architecture experiments

The `paper_reproduction/` directory contains architecture components and experimental model implementations.

Run the architecture smoke test:

```bash
python paper_reproduction/s2head_detection_smoke.py --device cpu
```

Use `--device mps` on a compatible Apple Silicon system.

The smoke test checks implementation and data compatibility; it is not a full training or accuracy reproduction.

The file `s2head_reconstruction.yaml` documents reconstruction metadata. It is not a standalone standard Ultralytics model YAML.

## 10. Knowledge distillation experiments

The `headlab_kd/` package contains the experimental KD implementation.

The CLI supports:

```bash
python -m headlab_kd --help
```

A smoke test can be run with:

```bash
python -m headlab_kd smoke
```

A training continuation can be run with:

```bash
python -m headlab_kd train --mode kd
```

The KD continuation requires pre-existing local checkpoints at:

```text
results/clean_experiments/
├── p2_yolo11n_50ep_seed42/weights/best.pt
└── teacher_yolo11s_p2_50ep_seed42/weights/best.pt
```

These checkpoints are not included in the Git repository. The control mode is available using `--mode control`.

The implementation and experiment settings are experimental adaptations. They should not be interpreted as exact reproduction of the original paper's complete distillation method.

The separate `paper_reproduction/s2head_tafm_pilot.py` workflow also depends on previously generated model checkpoints and experiment metadata. It is not a fresh-clone, one-command experiment.

## 11. Existing experiment results

The clean 50-epoch experiments on the deduplicated validation split produced the following results:

| Model | Part | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLO11n baseline | A | 87.63% | 80.28% | 86.84% | 36.28% |
| YOLO11n baseline | B | 90.94% | 83.74% | 91.53% | 51.47% |
| YOLO11n P2 | A | 88.43% | 82.88% | 90.48% | 40.16% |
| YOLO11n P2 | B | 91.36% | 84.20% | 92.80% | 53.21% |

These are results from the existing local experiment reports, not results automatically generated by cloning this repository. Reproduction requires the same dataset version, split memberships, software environment, and training settings.

The experimental KD continuation should be evaluated separately from the baseline/P2 comparison. Do not assume KD improves accuracy without considering its matched control results.

## 12. Reproducibility and limitations

- Dataset images and annotations are not included.
- Model checkpoints and training outputs are not included.
- Exact split membership is represented by the committed image-list manifests.
- A compatible local copy of the dataset must be supplied.
- Some research experiments require checkpoints and metadata generated by earlier training stages.
- The S2Head-style architecture is a reconstruction/adaptation, not an official authors' implementation.
- Results can vary with hardware, software versions, random seeds, and training environment.

## 13. Citation

If you use this repository in academic work, cite the research paper that motivated the project and clearly identify which implementation and experimental components you used.

## 14. Contributors

This project was developed collaboratively by:

- **Ritik** — [@itsrtk17](https://github.com/itsrtk17)
- **Sahil Kumar** — [@Sahilk-code](https://github.com/Sahilk-code)

