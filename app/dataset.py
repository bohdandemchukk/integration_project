from pathlib import Path
from torchaudio.datasets import SPEECHCOMMANDS
from app.config import CLASSES, DATA_DIR
from app.features import waveform_to_mel


class SpeechSubset(SPEECHCOMMANDS):
    def __init__(self, subset: str):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(str(DATA_DIR), download=True, subset=subset)
        self._walker = [p for p in self._walker if Path(p).parent.name in CLASSES]

    def raw(self, n):
        waveform, sr, label, *_ = super().__getitem__(n)
        return waveform, label

    def __getitem__(self, n):
        waveform, label = self.raw(n)
        return waveform_to_mel(waveform), CLASSES.index(label)