import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.config import *
from app.dataset import SpeechSubset
from app.model import SmallCNN


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    torch.manual_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dl = DataLoader(SpeechSubset("training"), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
    val_dl = DataLoader(SpeechSubset("validation"), batch_size=BATCH_SIZE,
                        num_workers=2)

    model = SmallCNN(len(CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.numel()
        val_acc = accuracy(model, val_dl)
        print(f"Епоха {epoch}/{EPOCHS} | loss={running/len(train_dl.dataset):.4f}"
              f" | val_acc={val_acc*100:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"Готово. Найкраща val_acc = {best*100:.2f}%. Модель: {MODEL_PATH}")


if __name__ == "__main__":
    main()