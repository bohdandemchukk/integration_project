import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import DATA_DIR
from app.data import SpeechSubset
from app.evaluate import accuracy, evaluate
from app.model import SmallCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_dl = DataLoader(SpeechSubset("training", args.classes, args.data_dir),
                          batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(SpeechSubset("validation", args.classes, args.data_dir),
                        batch_size=args.batch_size, num_workers=2)

    model = SmallCNN(len(args.classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best, model_path = 0.0, out / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, disable=not sys.stderr.isatty()):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
        val_acc = accuracy(model, val_dl)
        print(f"Епоха {epoch}/{args.epochs} | loss={running / len(train_dl.dataset):.4f}"
              f" | val_acc={val_acc * 100:.2f}%", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    metrics = evaluate(model, model_path, args.classes, args.data_dir)
    metrics.update({
        "architecture": "SmallCNN (3 x Conv-BN-ReLU-MaxPool + GAP + FC)",
        "epochs": args.epochs,
        "seed": args.seed,
        "best_val_accuracy": round(best * 100, 2),
    })
    (out / "labels.json").write_text(json.dumps(args.classes, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()