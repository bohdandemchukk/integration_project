import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.config import ARTIFACTS_DIR, DATA_DIR
from app.data import SpeechSubset
from app.model import SmallCNN


def accuracy(model, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


def evaluate(model, model_path, classes, data_dir) -> dict:
    test_ds = SpeechSubset("testing", classes, data_dir)
    acc = accuracy(model, DataLoader(test_ds, batch_size=64)) * 100
    return {
        "accuracy": round(acc, 2),  # у відсотках
        "test_samples": len(test_ds),
        "model_size_kb": round(os.path.getsize(model_path) / 1024, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    art = Path(args.artifacts)
    classes = json.loads((art / "labels.json").read_text())
    torch.set_num_threads(1)
    model = SmallCNN(len(classes))
    model.load_state_dict(torch.load(art / "model.pt", map_location="cpu", weights_only=True))
    print(json.dumps(evaluate(model, art / "model.pt", classes, args.data_dir), indent=2))


if __name__ == "__main__":
    main()