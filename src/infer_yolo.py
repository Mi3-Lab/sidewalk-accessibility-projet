#!/usr/bin/env python3
"""
Infer with YOLO model on images.
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Infer with YOLO model.")
    parser.add_argument("--model", type=str, required=True, help="Trained model path")
    parser.add_argument("--source", type=str, required=True, help="Image or directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--project", type=str, default="runs/predict")
    parser.add_argument("--name", type=str, default="exp")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        project=args.project,
        name=args.name
    )
    for result in results:
        print(result)

if __name__ == "__main__":
    main()