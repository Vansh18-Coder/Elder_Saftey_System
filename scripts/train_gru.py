import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.gru_model import GRUFallClassifier, build_loader, evaluate_model, save_model, train_epoch


def load_partition(data_dir, name):
    x_path = Path(data_dir) / f"{name}_x.npz"
    y_path = Path(data_dir) / f"{name}_y.npz"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing partition files for {name} in {data_dir}.")

    x = np.load(x_path)["x"]
    y = np.load(y_path)["y"]
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Train a GRU-based fall detection model on pose sequences.")
    parser.add_argument("--data-dir", type=str, default="data/sequences/demo_dataset", help="Directory with train_X/train_y and val/test arrays.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="models/gru_demo")
    args = parser.parse_args()

    train_x, train_y = load_partition(args.data_dir, "train")
    val_x, val_y = load_partition(args.data_dir, "val")

    if train_x.size == 0:
        raise RuntimeError("Training data is empty. Add more pose sequences before training.")

    if val_x.size == 0:
        print("Warning: Validation set is empty. Training will continue without validation metrics.")
        val_loader = None
    else:
        val_loader = build_loader(val_x, val_y, batch_size=args.batch_size, shuffle=False)

    train_loader = build_loader(train_x, train_y, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUFallClassifier(
        input_dim=train_x.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        else:
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}")

    if val_loader is not None:
        final_val_loss, final_val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
    else:
        final_val_loss, final_val_acc = None, None

    config = {
        "input_dim": int(train_x.shape[-1]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "device": str(device),
        "val_loss": None if final_val_loss is None else float(final_val_loss),
        "val_accuracy": None if final_val_acc is None else float(final_val_acc),
    }

    save_model(model, args.output_dir, config)
    print(f"Saved trained model to {Path(args.output_dir).resolve()}\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
