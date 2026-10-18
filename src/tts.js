// Browser Silero v5_cis_base: predictor -> acoustic -> vocoder ONNX graphs.
// Tokenization and durations mirror PartTTSModelMultiAcc_v3 and the original forward method.
"use strict";

const N_FFT = 2400;

/** Inverse real FFT matrices for the vocoder: cos_t and sin_t [N_FFT, bins]. */
function irfftMatrices() {
  const bins = N_FFT / 2 + 1, cos = new Float32Array(N_FFT * bins), sin = new Float32Array(N_FFT * bins);
  for (let n = 0; n < N_FFT; n++) {
    for (let k = 0; k < bins; k++) {
      const w = k === 0 || k === bins - 1 ? 1 : 2, angle = (2 * Math.PI * k * n) / N_FFT;
      cos[n * bins + k] = (w * Math.cos(angle)) / N_FFT;
      sin[n * bins + k] = k === 0 || k === bins - 1 ? 0 : (w * Math.sin(angle)) / N_FFT;
    }
  }
  return { cos: new ort.Tensor("float32", cos, [N_FFT, bins]), sin: new ort.Tensor("float32", sin, [N_FFT, bins]) };
}

function roundEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

class SileroTTS {
  /** meta contains symbols, symbol_to_id, speakers, sos, eos; sessions contains the three TTS graphs. */
  constructor(meta, sessions) {
    this.meta = meta;
    this.sessions = sessions;
    this.allowed = new Set([...meta.symbols.slice(3), "^"]);
    this.matrices = irfftMatrices();
  }

  /** Russian token preparation: lowercase and retain only model symbols. */
  tokenize(text) {
    let t = text.toLowerCase().replaceAll("—", "–").replaceAll("‑", "-");
    t = [...t].filter((c) => this.allowed.has(c)).join("");
    t = t.replaceAll("^", " ^").replace(/\s+/g, " ").trim().replaceAll("^ ", "^").replace(/\^+/g, "^").trim();
    if (t.endsWith("^")) t = t.slice(0, -1).trim();
    return [...(this.meta.sos + t + this.meta.eos)].map((c) => this.meta.symbol_to_id[c]);
  }

  /** Map predicted token durations to frame indices at the original speaking rate. */
  static frameTokens(logDur) {
    const n = logDur.length, dur = new Float64Array(n);
    for (let i = 0; i < n; i++) dur[i] = roundEven(Math.max(Math.exp(logDur[i]) - 1, 0));
    dur[0] = Math.min(dur[0], 5); dur[n - 1] = Math.min(dur[n - 1], 7);
    dur[n - 2] = 13; dur[n - 3] = Math.min(dur[n - 3], 13);
    const frames = [];
    for (let i = 0; i < n; i++) for (let k = 0; k < Math.floor(dur[i] + 0.5); k++) frames.push(BigInt(i));
    return BigInt64Array.from(frames);
  }

  /** Apply the original pitch formula: p*k + (k-1)*mean_std_coef[speaker]; unvoiced zeros remain zero. */
  shiftPitch(pitch, speakerId, k) {
    if (k === 1) return pitch;
    const shift = (k - 1) * this.meta.mean_std_coef[speakerId];
    const data = Float32Array.from(pitch.data, (p) => (p === 0 ? 0 : p * k + shift));
    return new ort.Tensor("float32", data, pitch.dims);
  }

  /** Synthesize plus-marked text into float32 audio at 24 kHz; pitch=1 preserves the voice pitch. */
  async synthesize(text, voice, { pitch: k = 1 } = {}) {
    const tokens = this.tokenize(text), speakerId = this.meta.speakers[voice];
    const sequence = new ort.Tensor("int64", BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
    const speaker = new ort.Tensor("int64", BigInt64Array.from([BigInt(speakerId)]), [1]);
    const predicted = await this.sessions.predictor.run({ sequence, speaker });
    const pitch = this.shiftPitch(predicted.pitch, speakerId, k);
    const frames = SileroTTS.frameTokens(predicted.log_dur.data);
    const frameToken = new ort.Tensor("int64", frames, [frames.length]);
    const { mel } = await this.sessions.acoustic.run({ sequence, speaker, pitch, frame_token: frameToken });
    const { audio } = await this.sessions.vocoder.run({ mel, cos_t: this.matrices.cos, sin_t: this.matrices.sin });
    return audio.data;
  }
}

globalThis.SileroTTSLib = { SileroTTS };
