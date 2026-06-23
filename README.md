# MD-YOLO Ship Draft Measurement

This repository provides the code and released weights for MD-YOLO, a visual measurement framework for ship draft mark detection, waterline detection, draft-reading evaluation, and single-image visualization.

## 1. Data

To find the dataset used in this study, please make sure all files are downloaded from:

https://pan.baidu.com/s/1UeZzg4Yqc8Rt-j5UyatZAg

Extraction code: **please email at 2025710769@yangtzeu.edu.cn**

After downloading the dataset, update the `path` field in `ultralytics/cfg/datasets/waternum.yaml`.

Expected layout:

```text
dataset_root/
  images/
  labels/
  splits/
    train.txt
    val.txt
    test.txt
```

Class order:

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

## 2. Installation

Install a CUDA-enabled PyTorch version suitable for your environment, then run:

```bash
pip install -r requirements.txt
pip install -e .
```

## 3. Repository Structure

```text
scripts/
  train.py
  val.py
  predict.py
  evaluate.py
  visualize.py
ultralytics/
  cfg/datasets/waternum.yaml
  cfg/models/mdyolo/mdyolo.yaml
weights/
  mdyolo_b2.pt
  mdyolo_b16.pt
```

## 4. Training

```bash
python scripts/train.py \
  --data ultralytics/cfg/datasets/waternum.yaml \
  --model ultralytics/cfg/models/mdyolo/mdyolo.yaml \
  --epochs 300 \
  --imgsz 1024 \
  --batch 16 \
  --device 0
```

For limited GPU memory, reduce `--batch` and keep `--nbs 64`.

## 5. Validation

```bash
python scripts/val.py \
  --weights weights/mdyolo_b16.pt \
  --data ultralytics/cfg/datasets/waternum.yaml \
  --split test \
  --imgsz 1024 \
  --batch 4 \
  --device 0
```

## 6. Prediction

```bash
python scripts/predict.py \
  --weights weights/mdyolo_b16.pt \
  --source /path/to/images \
  --imgsz 1024 \
  --conf 0.25 \
  --device 0
```

## 7. Draft-Reading Evaluation

The test CSV should contain:

```text
image_name,manual_draft_m,scenarios
```

Run:

```bash
python scripts/evaluate.py \
  --model weights/mdyolo_b16.pt \
  --test-list /path/to/test.csv \
  --image-dir /path/to/images \
  --predictions-out outputs/predictions.csv \
  --verbose
```

## 8. Single-Image Visualization

```bash
python scripts/visualize.py \
  --model weights/mdyolo_b16.pt \
  --image /path/to/image.jpg \
  --output outputs/draft_visualization.jpg
```

Use `--show` to display the visualization window.

## 9. Weights

- `weights/mdyolo_b2.pt`: checkpoint trained with the small-batch setting.
- `weights/mdyolo_b16.pt`: checkpoint trained with the batch-16 setting.
