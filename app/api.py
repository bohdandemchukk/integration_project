import hashlib
import io
import json
import logging
import os
import time

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import ARTIFACTS_DIR, LOG_LEVEL, MODEL_PATH, MODEL_VERSION, NUM_THREADS, SAMPLE_RATE
from app.data import waveform_to_mel
from app.model import SmallCNN

logging.basicConfig(level=LOG_LEVEL.upper())
log = logging.getLogger("api")

app = FastAPI(title="Speech Commands API", version=MODEL_VERSION)

# модель завантажується один раз, під час старту
torch.set_num_threads(NUM_THREADS)
CLASSES = json.loads((ARTIFACTS_DIR / "labels.json").read_text())
METRICS = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
model = SmallCNN(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
MODEL_SHA = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
log.info("Model loaded: %s sha256=%s version=%s threads=%d",
         MODEL_PATH, MODEL_SHA, MODEL_VERSION, NUM_THREADS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "architecture": METRICS.get("architecture"),
        "classes": CLASSES,
        "parameters": METRICS.get("parameters"),
        "model_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 1),
        "accuracy": METRICS["accuracy"],  # у відсотках
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Порожній файл")
    try:
        waveform, sr = torchaudio.load(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Не вдалося прочитати WAV-файл")

    t0 = time.perf_counter()  # лише препроцесинг + інференс
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    mel = waveform_to_mel(waveform)
    with torch.no_grad():
        probs = torch.softmax(model(mel.unsqueeze(0)), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    return {
        "label": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        "latency_ms": round(latency_ms, 2),
        "model_version": MODEL_VERSION,
    }