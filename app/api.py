import io, time
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.config import *
from app.features import waveform_to_mel
from app.model import SmallCNN

app = FastAPI(title="Speech Commands API")

torch.set_num_threads(1)
model = SmallCNN(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    t0 = time.perf_counter()
    try:
        waveform, sr = torchaudio.load(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Не вдалося прочитати WAV-файл")
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    mel = waveform_to_mel(waveform)
    with torch.no_grad():
        probs = torch.softmax(model(mel.unsqueeze(0)), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    return {
        "prediction": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        "latency_ms": round(latency_ms, 2),
    }