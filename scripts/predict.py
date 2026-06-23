import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run MD-YOLO inference.")
    parser.add_argument("--weights", required=True, help="Trained checkpoint path.")
    parser.add_argument("--source", required=True, help="Image, directory, video, or stream source.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="Global NMS IoU threshold before task-adaptive handling.")
    parser.add_argument("--device", default=0, help="CUDA device or cpu.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "predict"), help="Output project directory.")
    parser.add_argument("--name", default="md-yolo", help="Run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format predictions.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
