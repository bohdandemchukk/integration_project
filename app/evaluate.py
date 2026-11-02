import json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from app.config import *
from app.dataset import SpeechSubset
from app.features import waveform_to_mel
from app.model import SmallCNN
from app.train import accuracy


def main():
    torch.set_num_threads(1)
    model = SmallCNN(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    test_ds = SpeechSubset("testing")

    acc = accuracy(model, DataLoader(test_ds, batch_size=64)) * 100

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    n_params = sum(p.numel() for p in model.parameters())

    N = min(200, len(test_ds))
    samples = [test_ds.raw(i)[0] for i in range(N)]
    times = []
    with torch.no_grad():
        for w in samples[:10]:
            model(waveform_to_mel(w).unsqueeze(0))
        for w in samples:
            t0 = time.perf_counter()
            mel = waveform_to_mel(w)
            model(mel.unsqueeze(0)).argmax(1)
            times.append((time.perf_counter() - t0) * 1000)
    lat_mean, lat_p95 = float(np.mean(times)), float(np.percentile(times, 95))

    res = {
        "dataset": "Google Speech Commands v0.02 (yes/no/up/down), аудіо",
        "model": "SmallCNN (3 x Conv-BN-ReLU-MaxPool + GAP + FC)",
        "parameters": n_params,
        "model_size_kb": round(size_kb, 1),
        "test_samples": len(test_ds),
        "accuracy_percent": round(acc, 2),
        "latency_mean_ms": round(lat_mean, 2),
        "latency_p95_ms": round(lat_p95, 2),
    }
    RESULTS_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()