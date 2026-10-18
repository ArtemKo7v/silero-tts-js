"""ONNX graph wrappers and duration processing for Silero v5_cis_base."""

import numpy as np
import torch

PHRASES = [
    "Ст+арый з+амок на холм+е был в+иден издалек+а.",
    "Поч+ему ты молч+ишь? +Ответь мн+е х+оть чт+о-ниб+удь!",
    "Здр+авствуйте, так звуч+ит +этот г+олос. Посл+ушайте, подх+одит ли он для в+ашей кн+иги.",
]
N_FFT, HOP = 2400, 600


def postprocess_durations(log_dur: np.ndarray) -> np.ndarray:
    """Token durations in frames, matching the original forward method at rate 1."""
    dur = np.exp(log_dur.astype(np.float32)) - 1
    dur[dur < 0] = 0
    dur = np.round(dur)
    dur[0, 0] = min(dur[0, 0], 5)
    dur[0, -1] = min(dur[0, -1], 7)
    dur[0, -2] = 13
    dur[0, -3] = min(dur[0, -3], 13)
    return dur


def frame_tokens(dur: np.ndarray) -> np.ndarray:
    """Return the token index per frame by repeating each token by its rounded duration."""
    repeats = np.floor(dur[0] + 0.5).astype(np.int64)
    return np.repeat(np.arange(dur.shape[1], dtype=np.int64), repeats)


class Predictor(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.dur, self.pitch = nets["dur"], nets["pitch"]

    def forward(self, sequence, speaker):
        log_dur = self.dur(sequence, speaker).squeeze(-1)
        pitch = self.pitch(sequence, speaker).transpose(1, 2)
        pitch = torch.where(pitch.abs() < 0.001, torch.zeros_like(pitch), pitch)
        return log_dur, pitch


class Vocoder(torch.nn.Module):
    def __init__(self, nets, jm):
        super().__init__()
        self.backbone, self.out = nets["backbone"], nets["head"]
        bins = N_FFT // 2 + 1
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(bins, dtype=torch.float64)
        angle = 2 * torch.pi * k[:, None] * n[None, :] / N_FFT
        weight = torch.full((bins, 1), 2.0, dtype=torch.float64)
        weight[0] = weight[-1] = 1.0
        # Inverse real FFT: imaginary DC and Nyquist components do not contribute.
        self.register_buffer("cos_t", (weight * torch.cos(angle) / N_FFT).T.float().contiguous())
        sin = weight * torch.sin(angle) / N_FFT
        sin[0] = sin[-1] = 0
        self.register_buffer("sin_t", sin.T.float().contiguous())
        self.register_buffer("window", jm.vocoder.head.istft.window.float().clone())
        self.register_buffer("H", jm.vocoder.pqmf_2.H.float().clone())

    @staticmethod
    def overlap_add(frames):
        """Overlap-add [N_FFT,T] frames at HOP spacing into a flat waveform."""
        t = frames.shape[1]
        parts = frames.T.reshape(t, N_FFT // HOP, HOP)
        total = None
        for r in range(N_FFT // HOP):
            shifted = torch.nn.functional.pad(parts[:, r, :], (0, 0, r, N_FFT // HOP - 1 - r))
            total = shifted if total is None else total + shifted
        return total.reshape(-1)

    def forward(self, mel):
        x = self.out(self.backbone(mel)).transpose(1, 2)[0]
        mag, phase = torch.chunk(x, 2, 0)
        mag = torch.clamp(torch.exp(mag), max=100.0)
        frames = (self.cos_t @ (mag * torch.cos(phase)) - self.sin_t @ (mag * torch.sin(phase))) * self.window[:, None]
        y = self.overlap_add(frames)
        envelope = self.overlap_add((self.window ** 2)[:, None].expand(-1, frames.shape[1]))
        pad = (N_FFT - HOP) // 2
        y = (y / envelope)[pad:-pad]
        bands = torch.nn.functional.conv1d(y[None, None], self.H, None, 2, 31)
        bands = torch.where(bands.abs().max() <= 1.0, bands, torch.tanh(bands))
        return bands[0, 0]
