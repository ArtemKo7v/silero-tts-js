// Russian text preparation: normalization, Cyrillic detection, and preservation of source stress.
// Includes sentence splitting for bounded model input.
"use strict";

const VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ";
const ACUTE = "́";
const ACUTE_RE = new RegExp(`([${VOWELS}])${ACUTE}`, "g");
const LONE_PLUS_RE = new RegExp(`\\+(?![${VOWELS}])`, "g");
const DECIMAL_RE = /(?<![\d.,])(\d+)[.,](\d{1,3})(?![\d.,]*\d)/g;
const NUMBER_RE = /\d{1,3}(?:[   ]\d{3})+(?!\d)|\d+/g;
const LATIN_RE = /[A-Za-z]+(?:'[A-Za-z]+)?/g;
const SYMBOLS = [["%", " процентов"], ["№", "номер "], ["=", " равно "], ["&", " и "]];
const UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь",
  "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
  "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"];
const TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
  "восемьдесят", "девяносто"];
const HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
  "восемьсот", "девятьсот"];
const SCALES = [["", "", "", false], ["тысяча", "тысячи", "тысяч", true],
  ["миллион", "миллиона", "миллионов", false], ["миллиард", "миллиарда", "миллиардов", false],
  ["триллион", "триллиона", "триллионов", false]];
const FRACTIONS = { 1: ["десятая", "десятых"], 2: ["сотая", "сотых"], 3: ["тысячная", "тысячных"] };
const LATIN_GROUPS = [["tion", "шн"], ["igh", "ай"], ["sh", "ш"], ["ch", "ч"], ["zh", "ж"], ["th", "з"],
  ["ph", "ф"], ["kh", "х"], ["ck", "к"], ["qu", "кв"], ["oo", "у"], ["ee", "и"], ["ea", "и"], ["ou", "ау"],
  ["ow", "оу"], ["ai", "эй"], ["ay", "эй"], ["ce", "се"], ["ci", "си"], ["cy", "си"]];
const LATIN_LETTERS = Object.fromEntries([..."abcdefghijklmnopqrstuvwxyz"].map((c, i) =>
  [c, ["а", "б", "к", "д", "е", "ф", "г", "х", "и", "дж", "к", "л", "м", "н", "о", "п", "к", "р", "с", "т",
    "у", "в", "в", "кс", "и", "з"][i]]));

function plural(number, one, few, many) {
  const n100 = Number(number % 100n);
  if (n100 >= 11 && n100 <= 14) return many;
  return { 1: one, 2: few, 3: few, 4: few }[Number(number % 10n)] ?? many;
}

function triplet(number, feminine) {
  const words = number >= 100 ? [HUNDREDS[Math.floor(number / 100)]] : [];
  let rest = number % 100;
  if (rest >= 20) { words.push(TENS[Math.floor(rest / 10)]); rest %= 10; }
  if (rest) words.push(feminine && (rest === 1 || rest === 2) ? (rest === 1 ? "одна" : "две") : UNITS[rest]);
  return words.join(" ");
}

/** Expand integer BigInt n below 10**15 in nominative case; larger values are read digit by digit. */
function numberWords(n, feminine = false) {
  if (n >= 10n ** 15n) return [...n.toString()].map((d) => UNITS[Number(d)]).join(" ");
  if (n === 0n) return UNITS[0];
  const words = [];
  for (let scale = SCALES.length - 1; scale >= 0; scale--) {
    const group = Number((n / 1000n ** BigInt(scale)) % 1000n);
    if (group === 0) continue;
    const [one, few, many, scaleFeminine] = SCALES[scale];
    words.push(triplet(group, scale ? scaleFeminine : feminine));
    if (scale) words.push(plural(BigInt(group), one, few, many));
  }
  return words.join(" ");
}

function decimalWords(whole, fraction) {
  const w = BigInt(whole), f = BigInt(fraction), [single, many] = FRACTIONS[fraction.length];
  return `${numberWords(w, true)} ${plural(w, "целая", "целых", "целых")} ` +
    `${numberWords(f, true)} ${plural(f, single, many, many)}`;
}

function transliterate(word) {
  const lower = word.toLowerCase().replaceAll("'", "");
  const result = [];
  let i = 0;
  while (i < lower.length) {
    const group = LATIN_GROUPS.find(([latin]) => lower.startsWith(latin, i));
    if (group) { result.push(group[1]); i += group[0].length; continue; }
    const c = lower[i];
    if (c === "y" && i === 0) result.push("й");
    else if (!(c === "e" && i === lower.length - 1 && lower.length > 3)) result.push(LATIN_LETTERS[c] ?? "");
    i += 1;
  }
  const text = result.join("") || word;
  const first = word.slice(0, 1);
  return first && first !== first.toLowerCase() ? text.slice(0, 1).toUpperCase() + text.slice(1) : text;
}

/** Normalize stress marks, symbols, Latin letters, and contextual numbers before synthesis. */
function normalizeForSilero(text) {
  text = text.replace(ACUTE_RE, "+$1").replaceAll(ACUTE, "");
  text = text.replace(LONE_PLUS_RE, " плюс ");
  for (const [symbol, words] of SYMBOLS) if (symbol !== "%") text = text.replaceAll(symbol, words);
  text = SileroNumbers.applyNumberRules(text);
  text = text.replaceAll("%", " процентов");
  text = text.replace(DECIMAL_RE, (_, whole, fraction) => decimalWords(whole, fraction));
  text = SileroNumbers.cardinalRule(text);
  text = text.replace(NUMBER_RE, (m) => numberWords(BigInt(m.replace(/\D/g, ""))));
  text = text.replace(LATIN_RE, (m) => transliterate(m));
  return text.replace(/[ \t]{2,}/g, " ").trim();
}

const CYRILLIC_RANGES = [[0x0400, 0x052f], [0x1c80, 0x1c8f], [0x2de0, 0x2dff], [0xa640, 0xa69f]];

/** Return the proportion of Cyrillic letters among all letters. */
function cyrillicRatio(text) {
  let letters = 0, cyrillic = 0;
  for (const char of text) {
    if (!/\p{L}/u.test(char)) continue;
    letters += 1;
    const code = char.codePointAt(0);
    if (CYRILLIC_RANGES.some(([low, high]) => code >= low && code <= high)) cyrillic += 1;
  }
  return letters ? cyrillic / letters : 0;
}

const PLUS_STRESS_RE = new RegExp(`\\+(?=[${VOWELS}])`, "g");
const ACUTE_STRESS_RE = new RegExp(`([${VOWELS}])${ACUTE}`, "g");
const MARKED_WORD_RE = new RegExp(`(?:[\\p{L}\\p{Nl}\\p{No}]|[+${ACUTE}])+`, "gu");

/** Keep words already stressed in source, converting acute accents to plus marks. */
function keepSourceStress(source, result) {
  const sourceWords = source.match(MARKED_WORD_RE) ?? [];
  if (sourceWords.length !== (result.match(MARKED_WORD_RE) ?? []).length) return result;
  let index = 0;
  return result.replace(MARKED_WORD_RE, (word) => {
    const original = sourceWords[index++];
    PLUS_STRESS_RE.lastIndex = 0; ACUTE_STRESS_RE.lastIndex = 0;
    if (!PLUS_STRESS_RE.test(original) && !ACUTE_STRESS_RE.test(original)) return word;
    return original.replace(ACUTE_STRESS_RE, "+$1");
  });
}

/** Split paragraphs into sentences; split long sentences at punctuation or spaces (at most maxLength). */
function splitSentences(text, maxLength = 300) {
  const paragraphs = text.split(/\n\s*\n|\r?\n/).map((p) => p.trim()).filter(Boolean);
  return paragraphs.map((paragraph) => {
    const sentences = paragraph.split(/(?<=[.!?…])\s+(?=\S)/).filter(Boolean);
    const result = [];
    for (let sentence of sentences) {
      while (sentence.length > maxLength) {
        const window = sentence.slice(0, maxLength);
        const cut = Math.max(window.lastIndexOf(", "), window.lastIndexOf("; "), window.lastIndexOf(" — "));
        const at = cut > maxLength / 3 ? cut + 1 : (window.lastIndexOf(" ") > 0 ? window.lastIndexOf(" ") : maxLength);
        result.push(sentence.slice(0, at).trim());
        sentence = sentence.slice(at).trim();
      }
      if (sentence) result.push(sentence);
    }
    return result;
  });
}

globalThis.SileroText = { normalizeForSilero, cyrillicRatio, keepSourceStress, splitSentences, numberWords };
