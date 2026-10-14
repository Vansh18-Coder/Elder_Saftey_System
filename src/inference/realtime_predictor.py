from __future__ import annotations

from collections import deque

import numpy as np
import torch

from src.training.gru_model import GRUFallClassifier, load_model


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
    def __init__(self, model_path, sequence_length=30, threshold=0.75):
        self.model = load_model(model_path, input_dim=132, hidden_dim=64, num_layers=2, num_classes=2)
        self.sequence_length = sequence_length
        self.buffer = SequenceBuffer(sequence_length=sequence_length)
        self.threshold = threshold

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
