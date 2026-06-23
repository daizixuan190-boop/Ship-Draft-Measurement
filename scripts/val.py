import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate or test an MD-YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Trained checkpoint path, e.g. runs/train/md-yolo/weights/best.pt.")
    parser.add_argument(
        "--data",
        default=str(ROOT / "ultralytics" / "cfg" / "datasets" / "waternum.yaml"),
        help="Dataset YAML path.",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=0, help="CUDA device or cpu.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "val"), help="Output project directory.")
    parser.add_argument("--name", default="md-yolo", help="Run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
