// Browser port of Silero Stress 1.5 (MIT, snakers4/silero-stress): accentor and homograph solver.
// Includes its BERT tokenizer and line-by-line stress wrapper.
// Defaults: stress words, resolve homographs, restore contextual yo, and skip monosyllabic stress.
// The n-gram EmbeddingBag and MLP run in JavaScript float32; homograph BERT runs in ONNX.
// export_stress.py supplies stress.json, stress.bin, and bert.onnx.
"use strict";

const STRESS_VOWELS = "аоуыэиеяёю";
const PLUS_GUARD = ""; // Hide literal source plus signs while the model runs.

function isLower(c) { return c !== c.toUpperCase() && c === c.toLowerCase(); }
function isUpper(c) { return c !== c.toLowerCase() && c === c.toUpperCase(); }
function countVowels(word) { let n = 0; for (const c of word) if (STRESS_VOWELS.includes(c)) n++; return n; }

/** Float32 dense layers: W[out, in] * x[in] + b; ReLU between layers. */
function mlp(layers, x) {
  for (let l = 0; l < layers.length; l++) {
    const { weight, bias, out, inp } = layers[l];
    const y = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = 0;
      const row = o * inp;
      for (let i = 0; i < inp; i++) s += weight[row + i] * x[i];
      y[o] = s + bias[o];
      if (l < layers.length - 1 && y[o] < 0) y[o] = 0;
    }
    x = y;
  }
  return x;
}

function softmaxArgmax(logits) {
  let max = -Infinity, arg = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) { max = logits[i]; arg = i; }
  let sum = 0;
  for (const v of logits) sum += Math.exp(v - max);
  return { arg, prob: 1 / sum }; // Argmax probability is exp(0) divided by the normalized exponential sum.
}

class Accentor {
  constructor(meta, weights) {
    this.ngrams = new Map(meta.ngrams.map((g, i) => [g, i]));
    this.unk = this.ngrams.get("UNK");
    this.exceptions = new Map(Object.entries(meta.exceptions));
    this.dim = meta.dim;
    this.emb = weights.acc_emb; // Int8Array [N, dim].
    this.scale = Math.fround(meta.scale); this.zero = Math.fround(meta.zero);
    const layers = (head) => [0, 2, 4, 6].map((i) => {
      const w = weights[`${head}.${i}.weight`];
      return { weight: w.data, bias: weights[`${head}.${i}.bias`].data, out: w.shape[0], inp: w.shape[1] };
    });
    this.stressClf = layers("stress_clf"); this.yoClf = layers("yo_clf");
  }

  /** Embed all known substrings of the boundary-marked word using their mean. */
  embed(word) {
    const text = `<${word}>`, ids = [];
    for (let n = 1; n <= word.length + 3; n++) {
      for (let i = 0; i + n <= text.length; i++) {
        const id = this.ngrams.get(text.slice(i, i + n));
        if (id !== undefined) ids.push(id);
      }
    }
    if (word.length < 1) { const id = this.ngrams.get(word); if (id !== undefined) ids.push(id); }
    if (!ids.length) ids.push(this.unk);
    const out = new Float32Array(this.dim);
    for (const id of ids) {
      for (let d = 0; d < this.dim; d++) out[d] += Math.fround(this.scale * (this.emb[id * this.dim + d] - this.zero));
    }
    for (let d = 0; d < this.dim; d++) out[d] = out[d] / ids.length;
    return out;
  }

  predict(word) {
    const e = this.embed(word);
    const stress = softmaxArgmax(mlp(this.stressClf, e)), yo = softmaxArgmax(mlp(this.yoClf, e));
    return { stressPred: stress.arg, stressProb: stress.prob, yoPred: yo.arg, yoProb: yo.prob };
  }

  tokenize(sentence) {
    const tokens = [], inputs = [], mask = [];
    for (const word of sentence.split(/([\s.,!?;:<>=()/\\]+)/)) {
      const parts = word.split("-");
      let curTokens, curMask;
      if (parts.length === 1) { curTokens = parts; curMask = [true]; }
      else {
        curTokens = [...parts.slice(0, -1).map((p) => p + "-"), parts.at(-1)];
        curMask = [...parts.slice(0, -1).map(() => true), parts.at(-1) !== "то"];
      }
      const curInputs = curTokens.map((t) => t.toLowerCase().replace(/[^А-Яа-яёЁ]/g, ""));
      tokens.push(...curTokens); inputs.push(...curInputs);
      mask.push(...curInputs.map((x, i) => x.length > 0 && curMask[i]));
    }
    return { tokens, inputs, mask };
  }

  positions(word, stressedIds, yoIds) {
    const vowelIds = [], yeIds = [];
    [...word].forEach((c, i) => { if (STRESS_VOWELS.includes(c)) vowelIds.push(i); if (c === "е") yeIds.push(i); });
    return {
      stress: stressedIds.filter((idx) => idx < vowelIds.length).map((idx) => vowelIds[idx]),
      yo: yoIds.filter((idx) => idx > 0 && idx - 1 < yeIds.length).map((idx) => yeIds[idx - 1]),
      numVowels: vowelIds.length, firstVowel: vowelIds.length ? vowelIds[0] : -1,
    };
  }

  exception(clean, raw, haveStress) {
    const [excStress, excYo] = this.exceptions.get(clean);
    const yo = (w, i) => w.slice(0, i) + (isLower(w[i]) ? "ё" : "Ё") + w.slice(i + 1);
    if (haveStress) {
      const userPositions = [...raw].map((c, i) => (c === "+" ? i : -1)).filter((i) => i >= 0);
      let word = raw.replaceAll("+", "");
      if (excYo !== -1 && userPositions.includes(excYo + 1)) word = yo(word, excYo);
      for (const pos of userPositions) word = word.slice(0, pos) + "+" + word.slice(pos);
      return word;
    }
    if (excYo !== -1) raw = yo(raw, excYo);
    return raw.slice(0, excStress) + "+" + raw.slice(excStress);
  }

  /** AccentorNgram call; homographs have already been resolved. */
  /** skipSingle excludes function words from the optional monosyllabic stress mode. */
  call(sentence, { putStress = true, putYo = true, singleVowel = false, skipSingle = null } = {}) {
    if (!(putStress || putYo)) return sentence;
    const { tokens, inputs, mask } = this.tokenize(sentence);
    const out = [];
    tokens.forEach((token, idx) => {
      let raw = token;
      const clean = inputs[idx], lower = raw.toLowerCase();
      if (!mask[idx]) { out.push(raw); return; }
      const haveStress = lower.includes("+"), haveYo = lower.includes("ё");
      if (haveStress && haveYo) { out.push(raw); return; }
      if (!haveStress && haveYo && putStress) {
        if (countVowels(lower) === 1 && !singleVowel) { out.push(raw); return; }
        const yoPositions = [...lower].map((c, i) => (c === "ё" ? i : -1)).filter((i) => i >= 0);
        yoPositions.forEach((pos, i) => { raw = raw.slice(0, pos + i) + "+" + raw.slice(pos + i); });
        out.push(raw); return;
      }
      if (this.exceptions.has(clean)) { out.push(this.exception(clean, raw, haveStress)); return; }
      const p = this.predict(clean);
      let stressedIds = [p.stressPred];
      const passedStress = p.stressProb > (putStress ? 0.5 : 1);
      let setStress = passedStress && !haveStress;
      const yoIds = [p.yoPred];
      const setYo = p.yoProb > (putYo ? 0.5 : 1);
      if (haveStress) stressedIds = lower.split("+").map((part) => countVowels(part));
      const pos = this.positions(lower, stressedIds, yoIds);
      if (pos.numVowels === 0) { out.push(raw); return; }
      for (const yoPos of pos.yo) {
        if (pos.stress.includes(yoPos) && setYo && lower[yoPos] === "е") {
          raw = raw.slice(0, yoPos) + (isLower(raw[yoPos]) ? "ё" : "Ё") + raw.slice(yoPos + 1);
        }
      }
      let stressPositions = pos.stress;
      if (pos.numVowels === 1) {
        stressPositions = [pos.firstVowel];
        setStress = singleVowel && putStress && !(skipSingle && skipSingle.has(clean));
      }
      if (!haveStress && setStress) {
        stressPositions.forEach((sp, i) => { raw = raw.slice(0, sp + i) + "+" + raw.slice(sp + i); });
      }
      out.push(raw);
    });
    return out.join("");
  }
}

// BERT BasicTokenizer and WordPiece without lowercasing, matching the upstream tokenizer.
class BertTokenizer {
  constructor(meta) {
    this.vocab = new Map(meta.vocab.map((t, i) => [t, i]).filter(([t]) => t !== ""));
    this.neverSplit = new Set(meta.never_split);
    Object.assign(this, { unk: meta.unk, cls: meta.cls, sep: meta.sep, pad: meta.pad,
                          homoStart: meta.homo_start, homoEnd: meta.homo_end });
  }

  static isPunctuation(c) {
    const cp = c.codePointAt(0);
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
      (cp >= 123 && cp <= 126) || /\p{P}/u.test(c);
  }

  static isChinese(cp) {
    return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) ||
      (cp >= 0x2a700 && cp <= 0x2b73f) || (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) ||
      (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
  }

  basic(text) {
    let clean = "";
    for (const c of text) {
      const cp = c.codePointAt(0);
      if (cp === 0 || cp === 0xfffd || (/\p{C}/u.test(c) && !"\t\n\r".includes(c))) continue;
      if (" \t\n\r".includes(c) || /\p{Zs}/u.test(c)) clean += " ";
      else clean += BertTokenizer.isChinese(cp) ? ` ${c} ` : c;
    }
    const out = [];
    for (const token of clean.trim().split(/\s+/).filter(Boolean)) {
      if (this.neverSplit.has(token)) { out.push(token); continue; }
      let start = true;
      const parts = [];
      for (const c of token) {
        if (BertTokenizer.isPunctuation(c)) { parts.push(c); start = true; }
        else { if (start) parts.push(""); start = false; parts[parts.length - 1] += c; }
      }
      out.push(...parts);
    }
    return out.join(" ").split(/\s+/).filter(Boolean);
  }

  wordpiece(token) {
    const chars = [...token];
    if (chars.length > 100) return ["[UNK]"];
    const pieces = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length, found = null;
      while (start < end) {
        let sub = chars.slice(start, end).join("");
        if (start > 0) sub = "##" + sub;
        if (this.vocab.has(sub)) { found = sub; break; }
        end -= 1;
      }
      if (found === null) return ["[UNK]"];
      pieces.push(found);
      start = end;
    }
    return pieces;
  }

  encode(text) {
    const tokens = this.basic(text).flatMap((t) => this.wordpiece(t));
    return [this.cls, ...tokens.map((t) => this.vocab.get(t) ?? this.unk), this.sep];
  }
}

class HomoSolver {
  constructor(meta, bertMeta, weights, bert) {
    this.homodict = new Map(Object.entries(meta.homodict));
    this.phrases = new Map(Object.entries(meta.phrases).map(([w, p]) =>
      [w, { re: new RegExp(p.source, "i"), variants: p.variants }]));
    this.window = meta.window;
    this.tokenizer = new BertTokenizer(bertMeta);
    this.bert = bert;
    this.clf = [0, 3].map((i) => {
      const w = weights[`homo_clf.${i}.weight`];
      return { weight: w.data, bias: weights[`homo_clf.${i}.bias`].data, out: w.shape[0], inp: w.shape[1] };
    });
  }

  static cleanText(text, isStart) {
    if (!text) return "";
    text = text.replace(/[^a-zA-Zа-яА-ЯёЁ0-9\s.!?,\-]/g, "").replace(/\s+/g, " ").replace(/-{2,}/g, " - ")
      .replace(/([.!?])\1+/g, "$1").replace(/,{2,}/g, ",").replace(/\s+([.,!?])/g, "$1")
      .replace(/([.!?])\1+/g, "$1").replace(/,{2,}/g, ",").replace(/([.,!?])(?=\S)/g, "$1 ")
      .replace(/\s+/g, " ").trim();
    if (isStart) {
      text = text.replace(/^[ .,!?-]+/, "");
      if (text) text = text[0].toUpperCase() + text.slice(1).toLowerCase();
    } else {
      if (text) text = text[0] + text.slice(1).toLowerCase();
      if (text && !".!?".includes(text.at(-1))) text += ".";
    }
    return text;
  }

  predictWithPattern(phrase, text) {
    const match = phrase.re.exec(text);
    if (!match) return null;
    for (let i = 0; i < phrase.variants.length; i++) {
      if (match.groups[`g${i}`] !== undefined) return phrase.variants[i];
    }
    return null;
  }

  async call(sentence, { putStress = true, putYo = true, singleVowel = false } = {}) {
    if (!(putStress || putYo)) return sentence;
    const items = [], batch = [];
    for (const match of sentence.matchAll(/(?=.*[а-яё])[а-яё+]+/gi)) {
      const word = match[0], lower = word.toLowerCase(), start = match.index, end = start + word.length;
      if (!this.homodict.has(lower) && !this.phrases.has(lower)) continue;
      const half = Math.floor(this.window / 2);
      const before = HomoSolver.cleanText(sentence.slice(0, start), true).slice(-half);
      const after = HomoSolver.cleanText(sentence.slice(end), false).slice(0, half);
      const rawMark = `${before} [HOMO] ${lower} [/HOMO] ${after}`.trim();
      let pred = this.phrases.has(lower) ? this.predictWithPattern(this.phrases.get(lower), rawMark) : null;
      if (pred === null) {
        if (!this.homodict.has(lower)) continue;
        const ids = this.tokenizer.encode(rawMark);
        batch.push({ ids, start: ids.indexOf(this.tokenizer.homoStart), end: ids.indexOf(this.tokenizer.homoEnd),
                     item: items.length });
      }
      items.push({ start, end, word, lower, pred });
    }
    if (!items.length) return sentence;
    if (batch.length) {
      // Match the original: batch each sentence and pad without an attention mask.
      const length = Math.max(...batch.map((b) => b.ids.length));
      const ids = new BigInt64Array(batch.length * length).fill(BigInt(this.tokenizer.pad));
      batch.forEach((b, i) => b.ids.forEach((id, j) => { ids[i * length + j] = BigInt(id); }));
      const { hidden } = await this.bert.run({ input_ids: new ort.Tensor("int64", ids, [batch.length, length]) });
      const dim = hidden.dims[2], h = hidden.data;
      batch.forEach((b, i) => {
        const x = new Float32Array(2 * dim), base = i * length * dim;
        for (let d = 0; d < dim; d++) x[d] = h[base + b.start * dim + d];
        const count = b.end - (b.start + 1);
        for (let t = b.start + 1; t < b.end; t++) for (let d = 0; d < dim; d++) x[dim + d] += h[base + t * dim + d];
        for (let d = 0; d < dim; d++) x[dim + d] = x[dim + d] / count;
        const logit = mlp(this.clf, x)[0];
        const p = 1 / (1 + Math.exp(-logit));
        const index = p > 0.5 ? 1 : 0; // torch.round uses ties to even: 0.5 -> 0.
        items[b.item].pred = this.homodict.get(items[b.item].lower)[index];
      });
    }
    let result = sentence, offset = 0;
    for (const { start, end, word, pred } of items) {
      let variant = putYo ? pred : pred.replaceAll("ё", "е");
      const nVowels = [...variant].filter((c) => STRESS_VOWELS.includes(c.toLowerCase())).length;
      const stressIdx = variant.indexOf("+");
      variant = variant.replace("+", "");
      variant = [...word].map((c1, i) => (i < variant.length ? (isLower(c1) ? variant[i].toLowerCase()
        : variant[i].toUpperCase()) : "")).join("");
      if ((nVowels > 1 || singleVowel) && putStress) {
        variant = variant.slice(0, stressIdx) + "+" + variant.slice(stressIdx);
      }
      result = result.slice(0, start + offset) + variant + result.slice(end + offset);
      if ((nVowels > 1 || singleVowel) && putStress) offset += 1;
    }
    return result;
  }
}

class SileroStress {
  /** meta is stress.json; weights maps names to data/shape; bert is the ONNX session. */
  constructor(meta, weights, bert) {
    this.accentor = new Accentor(meta.acc, weights);
    this.homosolver = new HomoSolver(meta.homo, meta.bert, weights, bert);
  }

  /** Process lines while guarding literal plus signs; singleVowel enables monosyllabic stress except skipSingle. */
  async stress(text, { singleVowel = false, skipSingle = null } = {}) {
    const out = [];
    for (const line of text.split("\n")) {
      if (!line.trim()) { out.push(line); continue; }
      const guarded = line.replaceAll("+", PLUS_GUARD);
      const homo = await this.homosolver.call(guarded, { singleVowel });
      out.push(this.accentor.call(homo, { singleVowel, skipSingle }).replaceAll(PLUS_GUARD, "+"));
    }
    return out.join("\n");
  }
}

// Unstressed monosyllabic prepositions, conjunctions, and particles.
const UNSTRESSED_WORDS = new Set(["а", "и", "но", "на", "по", "за", "до", "от", "из", "у", "о", "об", "под", "над",
  "без", "для", "про", "при", "со", "во", "ко", "не", "ни", "же", "ж", "ли", "ль", "бы", "б", "то", "ведь", "лишь"]);

/** Decode stress.bin using meta.weights into named TypedArrays and shapes. */
function readWeights(meta, buffer) {
  const result = {};
  for (const [name, w] of Object.entries(meta.weights)) {
    const Type = w.dtype === "int8" ? Int8Array : Float32Array;
    const data = new Type(buffer.slice(w.offset, w.offset + w.bytes));
    result[name] = name === "acc_emb" ? data : { data, shape: w.shape };
  }
  return result;
}

globalThis.SileroStressLib = { SileroStress, readWeights, UNSTRESSED_WORDS };
