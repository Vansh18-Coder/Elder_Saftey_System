import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import classification_report
from src.training.gru_model import GRUFallClassifier, load_model


def load_partition(data_dir, name):
    x_path = Path(data_dir) / f"{name}_x.npz"
    y_path = Path(data_dir) / f"{name}_y.npz"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {name} partition in {data_dir}.")
    x = np.load(x_path)["x"]
    y = np.load(y_path)["y"]
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GRU model on a held-out pose sequence dataset.")
    parser.add_argument("--model-path", type=str, default="models/gru_demo/gru_fall_classifier.pt", help="Path to the saved model checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data/sequences/demo_dataset", help="Directory containing dataset partitions.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Partition to evaluate on.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to save the metrics summary.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train a model first or pass --model-path.")

    data_dir = Path(args.data_dir)
    x, y = load_partition(data_dir, args.split)
    if x.size == 0:
        raise RuntimeError(f"Selected split '{args.split}' is empty in {data_dir}.")

    model = load_model(str(model_path), input_dim=int(x.shape[-1]), hidden_dim=64, num_layers=2, num_classes=2)
    model.eval()

    tensor_x = torch.tensor(x, dtype=torch.float32)
    logits = model(tensor_x)
    predictions = logits.argmax(dim=1).numpy()
    report = classification_report(y, predictions, num_classes=2)

    print(f"Evaluating model: {model_path}")
    print(f"Dataset split: {args.split}")
    print(f"Samples: {len(y)}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"F1: {report['f1']:.4f}")
    print(f"Specificity: {report['specificity']:.4f}")
    print("Confusion matrix:")
    print(report["confusion_matrix"])

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "samples": int(len(y)),
            "accuracy": float(report["accuracy"]),
            "precision": float(report["precision"]),
            "recall": float(report["recall"]),
            "f1": float(report["f1"]),
            "specificity": float(report["specificity"]),
            "confusion_matrix": report["confusion_matrix"].tolist(),
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
