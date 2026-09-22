from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs <= 0:
        raise ValueError("epochs must be positive")
    if arguments.batch_size <= 0:
        raise ValueError("batch size must be positive")

    model = YOLO(arguments.model)
    model.train(
        data=str(arguments.data),
        epochs=arguments.epochs,
        imgsz=arguments.image_size,
        batch=arguments.batch_size,
        device=arguments.device,
        workers=arguments.workers,
        seed=arguments.seed,
        deterministic=True,
        patience=arguments.patience,
        project=str(arguments.project.resolve()),
        name=arguments.name,
        exist_ok=False,
        pretrained=True,
        plots=True,
        verbose=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the acne-care YOLO11 baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", type=Path, default=Path("runs/baselines"))
    parser.add_argument("--name", default="yolo11n-acne-seed42")
    return parser.parse_args()


if __name__ == "__main__":
    main()
