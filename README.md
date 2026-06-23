# MD-YOLO Ship Draft Measurement

This repository contains the MD-YOLO code used for ship draft mark and waterline detection, draft-reading evaluation, and single-image visualization in fixed-berth ship draft measurement scenarios.

## Repository Structure

```text
.
├── scripts/
│   ├── train.py
│   ├── val.py
│   ├── predict.py
│   ├── evaluate.py
│   └── visualize.py
├── ultralytics/
│   ├── cfg/datasets/waternum.yaml
│   ├── cfg/models/mdyolo/mdyolo.yaml
│   └── ...
├── weights/
│   ├── mdyolo_b2.pt
│   └── mdyolo_b16.pt
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

Install a CUDA-enabled PyTorch build that matches your machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Format

Update `ultralytics/cfg/datasets/waternum.yaml` so that `path` points to your dataset root.

The expected YOLO-style split file layout is:

```text
dataset_root/
  images/
  labels/
  splits/
    train.txt
    val.txt
    test.txt
```

The class order must remain unchanged:

```text
0: 2
1: 4
2: 6
3: 8
4: M
5: 7
6: 9
7: 3
8: 1
9: 0
10: 5
11: waterline
```

## Training

```bash
python scripts/train.py \
  --data ultralytics/cfg/datasets/waternum.yaml \
  --model ultralytics/cfg/models/mdyolo/mdyolo.yaml \
  --epochs 300 \
  --imgsz 1024 \
  --batch 16 \
  --device 0
```

For limited GPU memory, reduce `--batch` and keep `--nbs 64` for nominal batch-size scaling.

## Detector Validation

Use `val.py` to reproduce detector-level metrics such as precision, recall, `mAP50`, and `mAP50-95`.

```bash
python scripts/val.py \
  --weights weights/mdyolo_b16.pt \
  --data ultralytics/cfg/datasets/waternum.yaml \
  --split test \
  --imgsz 1024 \
  --batch 4 \
  --device 0
```

## Detection Prediction

Use `predict.py` for ordinary YOLO-style detection visualization.

```bash
python scripts/predict.py \
  --weights weights/mdyolo_b16.pt \
  --source /path/to/images \
  --imgsz 1024 \
  --conf 0.25 \
  --device 0
```

## Draft-Reading Evaluation

Use `evaluate.py` to reconstruct physical draft readings and print scenario-wise metrological metrics.

The test CSV should contain:

```text
image_name,manual_draft_m,scenarios
```

Example:

```bash
python scripts/evaluate.py \
  --model weights/mdyolo_b16.pt \
  --test-list /path/to/test.csv \
  --image-dir /path/to/images \
  --predictions-out outputs/predictions.csv \
  --verbose
```

## Single-Image Visualization

Use `visualize.py` to estimate one image and save a visualized result.

```bash
python scripts/visualize.py \
  --model weights/mdyolo_b16.pt \
  --image /path/to/image.jpg \
  --output outputs/draft_visualization.jpg
```

Add `--show` to display the visualization in an OpenCV window.

## Weights

The `weights/` directory contains released MD-YOLO checkpoints:

- `mdyolo_b2.pt`: checkpoint corresponding to the small-batch training setting.
- `mdyolo_b16.pt`: checkpoint corresponding to the batch-16 training setting.

Use the checkpoint that matches the experiment you want to reproduce.
