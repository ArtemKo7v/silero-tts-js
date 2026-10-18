// Test monosyllabic stress with real Silero Stress weights.
// Requires models/stress.json and stress.bin; skipped when assets are absent.
// Run: node --test tests/*.test.cjs
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
require("../src/stress.js");

const models = path.join(__dirname, "..", "models");
const available = fs.existsSync(path.join(models, "stress.json")) && fs.existsSync(path.join(models, "stress.bin"));
const { SileroStress, readWeights, UNSTRESSED_WORDS } = globalThis.SileroStressLib;

test("односложные: «+» в значимых словах, служебные без «+»", { skip: !available && "нет models/" }, () => {
  const meta = JSON.parse(fs.readFileSync(path.join(models, "stress.json"), "utf8"));
  const buffer = fs.readFileSync(path.join(models, "stress.bin"));
  const weights = readWeights(meta, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
  const accentor = new SileroStress(meta, weights, null).accentor;
  const sentence = "Дождь шёл всю ночь, и он не стих до утра.";
  const cis = accentor.call(sentence, { singleVowel: true, skipSingle: UNSTRESSED_WORDS });
  for (const word of ["д+ождь", "ш+ёл", "н+очь", "+он"]) assert.ok(cis.toLowerCase().includes(word), `${word} в «${cis}»`);
  for (const word of [" и ", " не ", " до "]) assert.ok(cis.includes(word), `«${word.trim()}» без «+» в «${cis}»`);
  // The default parity mode omits stress on monosyllables.
  const service = accentor.call(sentence);
  assert.ok(service.includes("Дождь") && service.includes(" и "), service);
});
