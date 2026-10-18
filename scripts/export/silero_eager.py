"""Eager reconstruction of Silero v5_cis_base networks for ONNX export.

The original TorchScript attention fast paths and exceptions cannot be exported
directly. These modules reproduce its operations and parameter names, loading
weights with strict=True. The hourglass decoder deliberately uses the first
shortened layer input for its second layer, matching the original graph."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """ONNX-compatible batch-first multihead attention without masks."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, dim = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        shape = (b, length, self.heads, dim // self.heads)
        q, k, v = (t.reshape(shape).transpose(1, 2) for t in (q, k, v))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads), dim=-1)
        return self.out_proj((weights @ v).transpose(1, 2).reshape(b, length, dim))


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer("pe", torch.zeros(max_len, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,L,C] → [B,L,C]."""
        return x + (self.scale * self.pe[: x.shape[1]]).transpose(0, 1)


class FFTBlock(nn.Module):
    def __init__(self, dim: int, ff: int, heads: int = 2, kernel: int = 9) -> None:
        super().__init__()
        self.self_attn = Attention(dim, heads)
        self.conv1 = nn.Conv1d(dim, ff, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(ff, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x)).transpose(1, 2)
        x = x + self.conv2(torch.relu(self.conv1(x)))
        return self.norm2(x.transpose(1, 2))


class ForwardTransformer(nn.Module):
    def __init__(self, dim: int, ff: int, layers: int) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(dim)
        self.layers = nn.ModuleList(FFTBlock(dim, ff) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim: int, factor: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim * factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).reshape(x.shape[0], -1, self.dim)


class HourGlassTransformer(nn.Module):
    def __init__(self, dim: int, ff: int, factor: int = 3) -> None:
        super().__init__()
        self.factor = factor
        self.pos_encoder = PositionalEncoding(dim)
        self.pre_vanilla_layers = nn.ModuleList([FFTBlock(dim, ff)])
        self.shorten_layers = nn.ModuleList([FFTBlock(dim, ff), FFTBlock(dim, ff)])
        self.post_vanilla_layers = nn.ModuleList([FFTBlock(dim, ff)])
        self.upsample = LinearUpsample(dim, factor)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        x = self.pre_vanilla_layers[0](self.pos_encoder(x))
        remainder = (self.factor - length % self.factor) % self.factor
        x = F.pad(x, (0, 0, 0, remainder))
        downsampled = F.avg_pool1d(x.transpose(1, 2), self.factor, self.factor).transpose(1, 2)
        # The second shortened layer receives downsampled directly, matching the original.
        shortened = self.shorten_layers[1](downsampled)
        x = (self.upsample(shortened) + x)[:, :length]
        return self.norm(self.post_vanilla_layers[0](x))


class SeriesPredictor(nn.Module):
    """Per-token log-duration or pitch transformer predictor."""

    def __init__(self, symbols: int, speakers: int, emotions: int, dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(symbols, dim)
        self.speaker_embedding = nn.Embedding(speakers, dim)
        self.emotion_embedding = nn.Embedding(emotions, dim)  # Present in state_dict but unused in forward.
        self.transformer = ForwardTransformer(dim, dim, 4)
        self.lin = nn.Linear(dim, 1)

    def forward(self, sequence: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequence) + self.speaker_embedding(speaker).unsqueeze(1)
        return self.lin(self.transformer(x))  # [B,T,1]


class Tacotron(nn.Module):
    """FastPitch acoustic network: tokens, pitch, and frame indices -> mel [B,192,F]."""

    def __init__(self, symbols: int, speakers: int, dim: int = 128, mel: int = 192) -> None:
        super().__init__()
        self.embedding = nn.Embedding(symbols, dim)
        self.speaker_embedding = nn.Embedding(speakers, dim)
        self.encoder = ForwardTransformer(dim, 512, 4)
        self.pitch_proj = nn.Conv1d(1, dim, 3, padding=1)
        self.energy_proj = nn.Conv1d(1, dim, 3, padding=1)  # This model does not supply energy.
        self.decoder = HourGlassTransformer(dim, 512)
        self.lin = nn.Linear(dim, mel)
        self.pitch_strength = 1.0

    def forward(self, sequence, speaker, pitch, frame_token):
        x = self.encoder(self.embedding(sequence)) + self.speaker_embedding(speaker).unsqueeze(1)
        x = x + self.pitch_proj(pitch).transpose(1, 2) * self.pitch_strength
        x = torch.index_select(x, 1, frame_token)  # Duration regulator: map each frame to its token.
        return self.lin(self.decoder(x)).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 1536) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.gamma * self.pwconv2(F.gelu(self.pwconv1(y)))
        return x + y.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, mel: int = 192, dim: int = 512, blocks: int = 8) -> None:
        super().__init__()
        self.embed = nn.Conv1d(mel, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(ConvNeXtBlock(dim) for _ in range(blocks))
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.embed(mel).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))  # [B,F,512]


def load_from_jit(jm) -> dict:
    """Construct eager modules and load original submodule weights with strict=True."""
    symbols, speakers = jm.tacotron.embedding.weight.shape[0], jm.tacotron.speaker_embedding.weight.shape[0]
    def predictor(source):
        return SeriesPredictor(symbols, source.speaker_embedding.weight.shape[0],
                               source.emotion_embedding.weight.shape[0]), source

    modules = {
        "dur": predictor(jm.dur_predictor.dur_pred),
        "pitch": predictor(jm.pitch_predictor.pitch_pred),
        "tacotron": (Tacotron(symbols, speakers), jm.tacotron),
        "backbone": (VocosBackbone(), jm.vocoder.backbone),
        "head": (nn.Linear(512, 2402), jm.vocoder.head.out),
    }
    result = {}
    for name, (module, source) in modules.items():
        state = {k: v for k, v in source.state_dict().items()}
        if name == "tacotron":  # LengthRegulator has no weights; pitch_strength is an attribute.
            module.pitch_strength = float(jm.tacotron.pitch_strength)
        module.load_state_dict(state, strict=True)
        result[name] = module.eval()
    return result
