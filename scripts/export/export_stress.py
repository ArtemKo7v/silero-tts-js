"""Export Silero Stress 1.5 to browser dictionaries, binary weights, and ONNX BERT.

Usage: python -I scripts/export/export_stress.py OUT_DIR [--corpus CORPUS_TXT]
Writes stress.json, stress.bin, and bert.onnx; optional UTF-8 sentences produce
expected.json for browser/Python stress parity checks."""

import json
import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def stress_text(accentor, text: str) -> str:
    """Mirror the browser's default stress mode, protecting literal plus signs."""
    options = dict(put_stress=True, put_stress_homo=True, put_yo=True, put_yo_homo=True,
                   stress_single_vowel=False, words_to_ignore=None)
    guard = "\ue000"
    return "\n".join(accentor(line.replace("+", guard), **options).replace(guard, "+")
                     if line.strip() else line for line in text.split("\n"))


def reference_stress(accentor, text: str) -> str:
    """Preserve source stress and skip text below the browser's Cyrillic threshold."""
    ranges = ((0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))
    letters = [c for c in text if c.isalpha()]
    ratio = sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters) / len(letters) if letters else 0
    if ratio < 0.6:
        return text
    result = stress_text(accentor, text)
    pattern = r"(?:[^\W\d_]|[+\u0301])+"
    source = re.findall(pattern, text)
    if len(source) != len(re.findall(pattern, result)):
        return result
    words = iter(source)

    def restore(match):
        original = next(words)
        vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        if re.search(r"\+(?=[" + vowels + r"])|[" + vowels + "]\u0301", original):
            return re.sub(r"([" + vowels + "])\u0301", r"+\1", original)
        return match.group()

    return re.sub(pattern, restore, result)


class Bert(nn.Module):
    """Homograph BERT without attention masks; parameter names match TorchScript."""

    def __init__(self, vocab: int, dim: int = 312, layers: int = 3, heads: int = 12, inter: int = 600,
                 positions: int = 2048) -> None:
        super().__init__()
        self.heads = heads
        self.register_buffer("word_q", torch.zeros(vocab, dim, dtype=torch.int8))
        self.register_buffer("word_scale", torch.zeros(()))
        self.register_buffer("word_zero", torch.zeros(()))
        self.embeddings = nn.ModuleDict({
            "position_embeddings": nn.Embedding(positions, dim), "token_type_embeddings": nn.Embedding(2, dim),
            "LayerNorm": nn.LayerNorm(dim, eps=1e-12)})

        def layer():
            return nn.ModuleDict({
                "attention": nn.ModuleDict({
                    "self": nn.ModuleDict({k: nn.Linear(dim, dim) for k in ("query", "key", "value")}),
                    "output": nn.ModuleDict({"dense": nn.Linear(dim, dim), "LayerNorm": nn.LayerNorm(dim, eps=1e-12)})}),
                "intermediate": nn.ModuleDict({"dense": nn.Linear(dim, inter)}),
                "output": nn.ModuleDict({"dense": nn.Linear(inter, dim), "LayerNorm": nn.LayerNorm(dim, eps=1e-12)})})
        self.encoder = nn.ModuleDict({"layer": nn.ModuleList(layer() for _ in range(layers))})

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, length = input_ids.shape
        e = self.embeddings
        x = (self.word_q[input_ids].float() - self.word_zero) * self.word_scale
        x = x + e["token_type_embeddings"].weight[0] + e["position_embeddings"].weight[:length]
        x = e["LayerNorm"](x)
        dim = x.shape[-1]
        for layer in self.encoder["layer"]:
            a = layer["attention"]
            q, k, v = (a["self"][n](x).reshape(b, length, self.heads, dim // self.heads).transpose(1, 2)
                       for n in ("query", "key", "value"))
            weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads), dim=-1)
            context = (weights @ v).transpose(1, 2).reshape(b, length, dim)
            x = a["output"]["LayerNorm"](a["output"]["dense"](context) + x)
            h = F.gelu(layer["intermediate"]["dense"](x))
            x = layer["output"]["LayerNorm"](layer["output"]["dense"](h) + x)
        return x


def js_pattern(pattern: str) -> tuple[str, list[str]]:
    """Convert Python named regex groups to JS gN groups, retaining stress variants."""
    names = []

    def group(match):
        names.append(match.group(1))
        return f"(?<g{len(names) - 1}>"
    return re.sub(r"\(\?P<([^>]+)>", group, pattern), names


def decapitalize_stress(word: str) -> str:
    for index, char in enumerate(word):
        if char.isupper():
            return word[:index] + "+" + word[index:].lower()
    return word


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Silero Stress 1.5 weights, dictionaries, and BERT.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--corpus", type=Path, help="Optional UTF-8 reference sentences, one per line")
    args = parser.parse_args()
    corpus, out = args.corpus, args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    from silero_stress import load_accentor

    stresser = load_accentor()
    accentor, homo = stresser.accentor, stresser.homosolver

    blobs, offsets = [], {}

    def add(name: str, array: np.ndarray) -> None:
        start = sum(len(b) for b in blobs)
        data = np.ascontiguousarray(array).tobytes()
        blobs.append(data)
        offsets[name] = {"offset": start, "bytes": len(data), "dtype": str(array.dtype), "shape": list(array.shape)}

    # Recover the original int8 embeddings from scale*(q-zero_point).
    model = accentor.model
    scale, zero = float(model.scale), float(model.zero_point)
    weight = model.embedding.weight.detach().numpy()
    q = np.round(weight / scale + zero)
    assert np.abs(scale * (q - zero) - weight).max() < 1e-6, "Accentor embeddings are not int8"
    add("acc_emb", q.astype(np.int8))
    for head in ("stress_clf", "yo_clf"):
        for index in (0, 2, 4, 6):
            linear = getattr(getattr(model, head), str(index))
            add(f"{head}.{index}.weight", linear.weight.detach().numpy().astype(np.float32))
            add(f"{head}.{index}.bias", linear.bias.detach().numpy().astype(np.float32))
    ngram_dict = model.embedding.ngram_dict
    ngrams = [None] * len(ngram_dict)
    for gram, index in ngram_dict.items():
        ngrams[index] = gram
    assert None not in ngrams

    # Homograph classifier and BERT.
    for index in (0, 3):
        linear = getattr(homo.model.homo_clf, str(index))
        add(f"homo_clf.{index}.weight", linear.weight.detach().numpy().astype(np.float32))
        add(f"homo_clf.{index}.bias", linear.bias.detach().numpy().astype(np.float32))
    jit_bert = homo.model.bert
    word = jit_bert.embeddings.word_embeddings.weight.detach()
    bscale, bzero = float(jit_bert.scale), float(jit_bert.zero_point)
    bq = torch.round(word / bscale + bzero)
    assert float((bscale * (bq - bzero) - word).abs().max()) < 1e-5, "BERT embeddings are not int8"
    bert = Bert(word.shape[0])
    state = {k: v for k, v in jit_bert.state_dict().items()
             if not k.startswith(("embeddings.word_embeddings", "pooler", "embeddings.position_ids",
                                  "embeddings.token_type_ids")) and k not in ("scale", "zero_point")}
    state.update(word_q=bq.to(torch.int8), word_scale=torch.tensor(bscale), word_zero=torch.tensor(bzero))
    bert.load_state_dict(state, strict=True)
    bert.eval()
    ids = torch.randint(5, word.shape[0], (3, 40))
    diff = float((bert(ids) - jit_bert(ids)[0]).abs().max())
    print("BERT eager ↔ jit max diff", diff, flush=True)
    assert diff < 1e-4
    torch.onnx.export(bert, (ids,), str(out / "bert.onnx"), input_names=["input_ids"], output_names=["hidden"],
                      dynamic_axes={"input_ids": {0: "B", 1: "L"}, "hidden": {0: "B", 1: "L"}},
                      opset_version=17, dynamo=False)

    phrases = {}
    for key, compiled in homo.compiled_phrases.items():
        source, names = js_pattern(compiled.pattern)
        phrases[key] = {"source": source, "variants": [decapitalize_stress(n) for n in names]}
    tokenizer = homo.tokenizer
    vocab = [None] * (max(tokenizer.vocab.values()) + 1)
    for token, index in tokenizer.vocab.items():
        vocab[index] = token
    meta = {
        "acc": {"scale": scale, "zero": zero, "dim": int(weight.shape[1]), "ngrams": ngrams,
                "exceptions": {k: list(v) for k, v in accentor.exceptions.items()}},
        "homo": {"homodict": {k: sorted(v) for k, v in homo.homodict.items()}, "phrases": phrases,
                 "window": homo.window_size},
        "bert": {"vocab": [t if t is not None else "" for t in vocab], "never_split": sorted(tokenizer.never_split),
                 "unk": tokenizer.unk_token_id, "cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id,
                 "pad": tokenizer.pad_token_id, "homo_start": tokenizer.homo_start_id,
                 "homo_end": tokenizer.homo_end_id},
        "weights": offsets,
    }
    (out / "stress.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (out / "stress.bin").write_bytes(b"".join(blobs))

    expected = []
    for line in corpus.read_text(encoding="utf-8").splitlines() if corpus else []:
        if not line.strip():
            continue
        expected.append({"text": line, "stressed": reference_stress(stresser, line)})
    if corpus:
        (out / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=0), encoding="utf-8")
    print("Reference corpus:", len(expected), "sentences; file sizes:",
          {p.name: p.stat().st_size for p in out.iterdir()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
