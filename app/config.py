import os
from pathlib import Path

CLASSES = ["yes", "no", "up", "down"]
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
N_MELS = 40

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pt"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_DIR = MODEL_PATH.parent
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", str(MODEL_DIR / "results.json")))
ARTIFACTS_DIR = MODEL_PATH.parent

EPOCHS = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LR = float(os.getenv("LR", "1e-3"))

NUM_THREADS = int(os.getenv("NUM_THREADS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
