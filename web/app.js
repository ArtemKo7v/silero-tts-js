// Shared demo: load separate or embedded assets, normalize text, add stress, and synthesize WAV.
"use strict";

const SAMPLE_RATE = 24000;
const PARAGRAPH_PAUSE = 1.0; // Silence between paragraphs, in seconds.
const $ = (id) => document.getElementById(id);
let tts = null, stresser = null, stopRequested = false, audioUrl = null;
let manifest = null;

async function checkedFetch(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return response;
}

function hasPayload(name) {
  return Boolean(document.querySelector(`script[data-name="${name}"]`) || manifest?.[name]);
}

async function payload(name) {
  const node = document.querySelector(`script[data-name="${name}"]`);
  if (!node) {
    if (!manifest?.[name]) throw new Error(`Missing asset: ${name}`);
    return (await checkedFetch(manifest[name])).arrayBuffer();
  }
  const response = await fetch(`data:application/octet-stream;base64,${node.textContent.trim()}`);
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

async function payloadJson(name) {
  return JSON.parse(new TextDecoder().decode(await payload(name)));
}

function setStatus(text) { $("status").textContent = text; }

async function load() {
  const started = performance.now();
  if (!document.querySelector('script[data-name="ort-mjs"]')) {
    if (location.protocol === "file:") throw new Error("Serve the web build over HTTP, or open the standalone silero.html.");
    manifest = await (await checkedFetch("assets.json")).json();
  }
  setStatus("Loading ONNX Runtime…");
  // Blob URLs let Emscripten resolve both assets in an offline file without relative URL lookup.
  const glue = new Blob([await payload("ort-mjs")], { type: "text/javascript" });
  const wasm = new Blob([await payload("ort-wasm")], { type: "application/wasm" });
  ort.env.wasm.wasmPaths = { mjs: URL.createObjectURL(glue), wasm: URL.createObjectURL(wasm) };
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
  const options = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const sessions = {};
  for (const name of ["predictor", "acoustic", "vocoder", "bert"]) {
    setStatus(`Loading ${name} model…`);
    sessions[name] = await ort.InferenceSession.create(await payload(name), options);
  }
  URL.revokeObjectURL(ort.env.wasm.wasmPaths.mjs);
  URL.revokeObjectURL(ort.env.wasm.wasmPaths.wasm);
  setStatus("Loading stress dictionaries…");
  const meta = await payloadJson("stress-json");
  const weights = SileroStressLib.readWeights(meta, await payload("stress-bin"));
  stresser = new SileroStressLib.SileroStress(meta, weights, sessions.bert);
  tts = new SileroTTSLib.SileroTTS(await payloadJson("tts-meta"), sessions);
  const voices = await payloadJson("voices");
  for (const [id, name, gender] of voices) {
    $("voice").append(new Option(`${name} (${gender})`, id, id === "ru_aigul", id === "ru_aigul"));
  }
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  setStatus(`Ready: models loaded in ${seconds} s; WASM threads: ${ort.env.wasm.numThreads}.`);
  $("speak").disabled = false;
  if (hasPayload("expected")) $("selftest").hidden = false;
}

/** Stress text with at least 60% Cyrillic letters, preserving user-supplied stress. */
async function stressLikeService(text) {
  if (SileroText.cyrillicRatio(text) < 0.6) return text;
  return SileroText.keepSourceStress(text, await stresser.stress(text));
}

/** CIS voices also need stress on meaningful monosyllables, excluding function words. */
async function stressForCis(text) {
  if (SileroText.cyrillicRatio(text) < 0.6) return text;
  const options = { singleVowel: true, skipSingle: SileroStressLib.UNSTRESSED_WORDS };
  return SileroText.keepSourceStress(text, await stresser.stress(text, options));
}

/** Normalize before adding stress so expanded numbers and Latin letters receive stress too. */
async function prepare(sentence) {
  const normalized = SileroText.normalizeForSilero(sentence);
  return { normalized, final: await stressForCis(normalized) };
}

function wav(samples) {
  const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true); view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function speak() {
  const voice = $("voice").value, paragraphs = SileroText.splitSentences($("text").value);
  const pitch = Number($("pitch").value); // Frequency multiplier; 1 keeps the original pitch.
  const total = paragraphs.reduce((n, p) => n + p.length, 0);
  if (!total) return;
  $("speak").disabled = true; $("stop").disabled = false; stopRequested = false;
  $("log").replaceChildren(); $("result").hidden = true;
  $("progress").value = 0;
  const chunks = [], started = performance.now();
  let done = 0, audioSeconds = 0, failures = 0;
  for (const [p, paragraph] of paragraphs.entries()) {
    if (p > 0) chunks.push(new Float32Array(Math.round(PARAGRAPH_PAUSE * SAMPLE_RATE)));
    for (const sentence of paragraph) {
      if (stopRequested) break;
      const row = document.createElement("li");
      try {
        const { final } = await prepare(sentence);
        row.textContent = final;
        if (/[а-яё]/i.test(final)) {
          const audio = await tts.synthesize(final, voice, { pitch });
          chunks.push(audio);
          audioSeconds += audio.length / SAMPLE_RATE;
        } else {
          row.className = "muted"; row.textContent = `${sentence} — skipped: no Russian text to read`;
        }
      } catch (error) {
        failures += 1;
        row.className = "error"; row.textContent = `${sentence} — error: ${error.message ?? error}`;
      }
      $("log").append(row);
      done += 1;
      const elapsed = (performance.now() - started) / 1000;
      $("progress").value = done / total;
      setStatus(`Sentence ${done} of ${total}; ${audioSeconds.toFixed(1)} s of audio in ${elapsed.toFixed(1)} s` +
        (audioSeconds ? ` (RTF ${(elapsed / audioSeconds).toFixed(2)})` : ""));
      await new Promise((r) => setTimeout(r)); // Let the browser update progress.
    }
    if (stopRequested) break;
  }
  const length = chunks.reduce((n, c) => n + c.length, 0), all = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) { all.set(c, offset); offset += c.length; }
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(wav(all));
  $("audio").src = audioUrl;
  $("download").href = audioUrl;
  const level = $("pitch").selectedOptions[0].dataset.level;
  $("download").download = `silero-${voice}${level === "0" ? "" : `-pitch${level}`}.wav`;
  $("result").hidden = length === 0;
  $("speak").disabled = false; $("stop").disabled = true;
  if (failures) setStatus(`${failures} sentence(s) failed; see the text log below.`);
  else if (stopRequested) setStatus(`Stopped after ${done} of ${total} sentences.`);
  return { sentences: done, failures, samples: length };
}

/** Compare JS stress with the optional Python reference corpus from export_stress.py. */
async function selftest() {
  $("selftest").disabled = true;
  const expected = await payloadJson("expected"), mismatches = [];
  const started = performance.now();
  for (const [i, item] of expected.entries()) {
    const stressed = await stressLikeService(item.text);
    if (stressed !== item.stressed) mismatches.push({ i, text: item.text, js: stressed, py: item.stressed });
    if (i % 20 === 0) { setStatus(`Checking: ${i} of ${expected.length}`); await new Promise((r) => setTimeout(r)); }
  }
  const result = { total: expected.length, mismatches: mismatches.length,
                   seconds: +((performance.now() - started) / 1000).toFixed(1), examples: mismatches.slice(0, 30) };
  $("selftest-result").textContent = JSON.stringify(result, null, 1);
  setStatus(`Parity check: ${expected.length - mismatches.length} of ${expected.length} match`);
  $("selftest").disabled = false;
  return result;
}

$("speak").onclick = () => speak().catch((e) => {
  setStatus("Error: " + e); $("speak").disabled = false; $("stop").disabled = true;
});
$("stop").onclick = () => { stopRequested = true; };
$("selftest").onclick = () => selftest().catch((e) => {
  setStatus("Parity check error: " + e); $("selftest").disabled = false;
});
/** #autotest runs parity (if available) and synthesis, reporting success in document.title. */
async function autotest() {
  const check = !$("selftest").hidden ? await selftest() : null;
  const synthesis = await speak();
  document.title = "AUTOTEST " + JSON.stringify({ ok: Boolean(synthesis?.samples) && synthesis.failures === 0 && !check?.mismatches,
    synthesis, check, status: $("status").textContent,
    audio: $("audio").src.startsWith("blob:"), isolated: self.crossOriginIsolated, href: location.protocol });
}

load().then(() => (location.hash === "#autotest" ? autotest() : null))
  .catch((e) => { setStatus("Loading failed: " + (e.message ?? e)); document.title = "AUTOTEST error " + e; });
