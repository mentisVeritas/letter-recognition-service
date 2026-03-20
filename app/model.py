import os
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "cnn.pth")

_model = None


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 26),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model() -> CNN:
    global _model

    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        _model = CNN()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.to("cpu")
        _model.eval()

    return _model