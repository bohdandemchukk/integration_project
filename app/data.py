from pathlib import Path

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from app.config import N_MELS, NUM_SAMPLES, SAMPLE_RATE

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=N_MELS
)
_to_db = torchaudio.transforms.AmplitudeToDB()


def waveform_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    n = waveform.shape[1]
    if n < NUM_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, NUM_SAMPLES - n))
    else:
        waveform = waveform[:, :NUM_SAMPLES]
    mel = _to_db(_mel(waveform))
    return (mel - mel.mean()) / (mel.std() + 1e-6)


class SpeechSubset(SPEECHCOMMANDS):
    def __init__(self, subset: str, classes, data_dir):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        super().__init__(str(data_dir), download=True, subset=subset)
        self.classes = list(classes)
        self._walker = [p for p in self._walker if Path(p).parent.name in self.classes]

    def raw(self, n):
        waveform, _sr, label, *_ = super().__getitem__(n)
        return waveform, label

    def __getitem__(self, n):
        waveform, label = self.raw(n)
        return waveform_to_mel(waveform), self.classes.index(label)