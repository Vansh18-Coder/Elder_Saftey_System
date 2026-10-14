from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GRUFallClassifier(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        dropped = self.dropout(last_step)
        return self.classifier(dropped)


def build_loader(x, y, batch_size=16, shuffle=True):
    x_tensor = torch.tensor(np.asarray(x, dtype=np.float32))
    y_tensor = torch.tensor(np.asarray(y, dtype=np.int64))
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_y.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_items += batch_y.size(0)

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    targets = []
    predictions = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * batch_y.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_items += batch_y.size(0)
        targets.extend(batch_y.cpu().tolist())
        predictions.extend(preds.cpu().tolist())

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1), targets, predictions


def save_model(model, output_dir, config):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "gru_fall_classifier.pt")
    with open(output_path / "gru_model_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return output_path


def load_model(model_path, input_dim=132, hidden_dim=64, num_layers=2, num_classes=2):
    model = GRUFallClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
