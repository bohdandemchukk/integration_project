from pathlib import Path

CLASSES = ["yes", "no", "up", "down"]
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
N_MELS = 40

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pt"
RESULTS_PATH = MODEL_DIR / "results.json"

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3