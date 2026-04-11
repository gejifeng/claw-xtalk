"""Minimal compatibility shim for CosyVoice's use of openai-whisper."""
import torch
import torchaudio

from .tokenizer import Tokenizer


def log_mel_spectrogram(audio, n_mels: int = 80):
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)
    audio = audio.float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        center=True,
        power=2.0,
        normalized=False,
    ).to(audio.device)
    mel = mel_transform(audio)
    return torch.clamp(mel, min=1e-10).log10()


__all__ = ["Tokenizer", "log_mel_spectrogram"]