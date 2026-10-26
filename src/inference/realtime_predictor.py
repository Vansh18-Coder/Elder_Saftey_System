from __future__ import annotations

from collections import deque

import numpy as np
import torch

from pathlib import Path
import json

from src.training.gru_model import load_model


class SequenceBuffer:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def add(self, feature_vector):
        if feature_vector is None:
            return False
        self.buffer.append(np.asarray(feature_vector, dtype=np.float32))
        return len(self.buffer) >= self.sequence_length

    def ready(self):
        return len(self.buffer) >= self.sequence_length

    def to_sequence(self):
        if not self.ready():
            return None
        return np.asarray(list(self.buffer), dtype=np.float32)


class RealTimePredictor:
    def __init__(self, model_path, sequence_length=30, threshold=None):
        config_path = Path(model_path).with_name("gru_model_config.json")
        model_config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                model_config = json.load(handle)

        self.model = load_model(
            model_path,
            input_dim=int(model_config.get("input_dim", 132)),
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            num_layers=int(model_config.get("num_layers", 2)),
            num_classes=2,
        )
        self.sequence_length = sequence_length
        self.buffer = SequenceBuffer(sequence_length=sequence_length)
        if threshold is None:
            threshold = model_config.get("fall_threshold")
            if threshold is None:
                threshold = 0.75
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Fall confidence threshold must be between 0 and 1.")
        self.threshold = float(threshold)

    def update(self, feature_vector):
        if feature_vector is None:
            return {"status": "NO_PERSON", "confidence": 0.0, "prediction": 0}

        if not self.buffer.add(feature_vector):
            return {"status": "COLLECTING", "confidence": 0.0, "prediction": 0}

        sequence = self.buffer.to_sequence()[None, :, :]
        with torch.no_grad():
            logits = self.model(torch.tensor(sequence, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[0]
            fall_probability = float(probs[1].item())
            prediction = 1 if fall_probability >= self.threshold else 0

        return {
            "status": "READY",
            "confidence": fall_probability,
            "prediction": prediction,
        }

    def reset(self):
        self.buffer = SequenceBuffer(sequence_length=self.sequence_length)
