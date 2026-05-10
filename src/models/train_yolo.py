#!/usr/bin/env python3
"""
Train YOLO model for sidewalk feature detection.
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model.")
    parser.add_argument("--model", type=str, default="yolo12n.pt", help="Model path or name")
    parser.add_argument("--data", type=str, required=True, help="Data YAML file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name
    )

if __name__ == "__main__":
    main()