import torch
import torchaudio
from app.config import SAMPLE_RATE, NUM_SAMPLES, N_MELS

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