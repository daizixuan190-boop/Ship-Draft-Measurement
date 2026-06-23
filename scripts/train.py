import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def build_weather_augmentations():
    import albumentations as A

    return [
        A.RandomRain(p=0.15, blur_value=3, brightness_coefficient=0.9, slant_range=(-10, 10)),
        A.RandomFog(fog_coef_range=(0.10, 0.25), alpha_coef=0.08, p=0.10),
        A.RandomShadow(p=0.15, num_shadows_limit=(1, 2), shadow_dimension=4, shadow_roi=(0.0, 0.5, 1.0, 1.0)),
        A.RandomBrightnessContrast(p=0.15, brightness_limit=0.15, contrast_limit=0.15),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Train MD-YOLO for ship draft measurement.")

    parser.add_argument(
        "--model",
        default=str(ROOT / "ultralytics" / "cfg" / "models" / "mdyolo" / "mdyolo.yaml"),
        help="Model YAML path.",
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "ultralytics" / "cfg" / "datasets" / "waternum.yaml"),
        help="Dataset YAML path.",
    )

    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Physical batch size.")
    parser.add_argument("--device", default=0, help="CUDA device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers.")

    parser.add_argument("--project", default=str(ROOT / "runs" / "train"), help="Output project directory.")
    parser.add_argument("--name", default="mdyolo", help="Run name.")

    parser.add_argument("--optimizer", default="AdamW", help="Optimizer.")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor.")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup momentum.")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1, help="Warmup bias LR.")

    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for loss scaling.")
    parser.add_argument("--close-mosaic", type=int, default=100, help="Disable mosaic during final N epochs.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp probability.")
    parser.add_argument("--cutmix", type=float, default=0.0, help="CutMix probability.")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste probability.")

    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain.")
    parser.add_argument("--cls", type=float, default=6.0, help="Classification loss gain.")
    parser.add_argument("--dfl", type=float, default=2.0, help="DFL loss gain.")

    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-H augmentation.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-S augmentation.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-V augmentation.")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation.")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation augmentation.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation.")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear augmentation.")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing probability.")

    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic training.")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained weights.")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP.")
    parser.add_argument("--weather-aug", action="store_true", help="Enable extra weather robustness augmentations.")

    return parser.parse_args()


def main():
    args = parse_args()

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        nbs=args.nbs,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cutmix=args.cutmix,
        copy_paste=args.copy_paste,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        erasing=args.erasing,
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
        pretrained=args.pretrained,
        amp=args.amp,
        cos_lr=False,
        save=True,
        val=True,
        split="val",
        plots=True,
    )

    if args.weather_aug:
        train_kwargs["augmentations"] = build_weather_augmentations()

    model = YOLO(args.model)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
