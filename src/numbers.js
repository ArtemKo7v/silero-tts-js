// Contextual Russian number expansion: inflect cardinal and ordinal numbers.
// Rules cover years, dates, suffixes, Roman numerals, percentages, and time.
// Case and gender use neighboring prepositions and noun endings, without a morphological dictionary.
// applyNumberRules runs first; text.js expands remaining numeric forms.
"use strict";

const CASE_INDEX = { nom: 0, gen: 1, dat: 2, acc: 3, ins: 4, prep: 5 };
const L = "А-Яа-яЁё";          // Cyrillic word boundaries without the Unicode flag.
const NOT_LETTER_BEFORE = `(?<![${L}A-Za-z])`;
const NOT_LETTER_AFTER = `(?![${L}A-Za-z+])`;
// Allow user-supplied plus stress marks between letters of a word.
const plusy = (alternatives) => alternatives.split("|").map((w) => [...w].join("\\+?")).join("|");

// Cardinals: nominative, genitive, dative, inanimate accusative, instrumental, prepositional.
const ONE = { m: ["один", "одного", "одному", "один", "одним", "одном"],
              f: ["одна", "одной", "одной", "одну", "одной", "одной"],
              n: ["одно", "одного", "одному", "одно", "одним", "одном"] };
const TWO = { m: ["два", "двух", "двум", "два", "двумя", "двух"], f: ["две", "двух", "двум", "две", "двумя", "двух"] };
TWO.n = TWO.m;
const soft = (nom, stem) => [nom, stem + "и", stem + "и", nom, nom.endsWith("ь") ? nom + "ю" : stem + "ью", stem + "и"];
const CARDINAL = {
  0: ["ноль", "ноля", "нолю", "ноль", "нолём", "ноле"],
  3: ["три", "трёх", "трём", "три", "тремя", "трёх"],
  4: ["четыре", "четырёх", "четырём", "четыре", "четырьмя", "четырёх"],
  5: soft("пять", "пят"), 6: soft("шесть", "шест"), 7: soft("семь", "сем"),
  8: ["восемь", "восьми", "восьми", "восемь", "восемью", "восьми"], 9: soft("девять", "девят"), 10: soft("десять", "десят"),
  11: soft("одиннадцать", "одиннадцат"), 12: soft("двенадцать", "двенадцат"), 13: soft("тринадцать", "тринадцат"),
  14: soft("четырнадцать", "четырнадцат"), 15: soft("пятнадцать", "пятнадцат"), 16: soft("шестнадцать", "шестнадцат"),
  17: soft("семнадцать", "семнадцат"), 18: soft("восемнадцать", "восемнадцат"), 19: soft("девятнадцать", "девятнадцат"),
  20: soft("двадцать", "двадцат"), 30: soft("тридцать", "тридцат"),
  40: ["сорок", "сорока", "сорока", "сорок", "сорока", "сорока"],
  50: ["пятьдесят", "пятидесяти", "пятидесяти", "пятьдесят", "пятьюдесятью", "пятидесяти"],
  60: ["шестьдесят", "шестидесяти", "шестидесяти", "шестьдесят", "шестьюдесятью", "шестидесяти"],
  70: ["семьдесят", "семидесяти", "семидесяти", "семьдесят", "семьюдесятью", "семидесяти"],
  80: ["восемьдесят", "восьмидесяти", "восьмидесяти", "восемьдесят", "восемьюдесятью", "восьмидесяти"],
  90: ["девяносто", "девяноста", "девяноста", "девяносто", "девяноста", "девяноста"],
  100: ["сто", "ста", "ста", "сто", "ста", "ста"],
  200: ["двести", "двухсот", "двумстам", "двести", "двумястами", "двухстах"],
  300: ["триста", "трёхсот", "трёмстам", "триста", "тремястами", "трёхстах"],
  400: ["четыреста", "четырёхсот", "четырёмстам", "четыреста", "четырьмястами", "четырёхстах"],
  500: ["пятьсот", "пятисот", "пятистам", "пятьсот", "пятьюстами", "пятистах"],
  600: ["шестьсот", "шестисот", "шестистам", "шестьсот", "шестьюстами", "шестистах"],
  700: ["семьсот", "семисот", "семистам", "семьсот", "семьюстами", "семистах"],
  800: ["восемьсот", "восьмисот", "восьмистам", "восемьсот", "восемьюстами", "восьмистах"],
  900: ["девятьсот", "девятисот", "девятистам", "девятьсот", "девятьюстами", "девятистах"],
};
// Scale nouns: singular/plural by case; 2-4 in nominative/accusative use genitive singular.
const SCALES = [
  null,
  { gender: "f", sg: ["тысяча", "тысячи", "тысяче", "тысячу", "тысячей", "тысяче"],
    pl: ["тысяч", "тысяч", "тысячам", "тысяч", "тысячами", "тысячах"], ordinal: "тысячн" },
  { gender: "m", sg: ["миллион", "миллиона", "миллиону", "миллион", "миллионом", "миллионе"],
    pl: ["миллионов", "миллионов", "миллионам", "миллионов", "миллионами", "миллионах"], ordinal: "миллионн" },
  { gender: "m", sg: ["миллиард", "миллиарда", "миллиарду", "миллиард", "миллиардом", "миллиарде"],
    pl: ["миллиардов", "миллиардов", "миллиардам", "миллиардов", "миллиардами", "миллиардах"], ordinal: "миллиардн" },
  { gender: "m", sg: ["триллион", "триллиона", "триллиону", "триллион", "триллионом", "триллионе"],
    pl: ["триллионов", "триллионов", "триллионам", "триллионов", "триллионами", "триллионах"], ordinal: "триллионн" },
];

/** Choose the noun form: one (1, 21...), few (2-4, 22...), many (5-20, 0...). */
function countClass(n) {
  const n100 = Number(BigInt(n) % 100n), n10 = n100 % 10;
  if (n100 >= 11 && n100 <= 14) return "many";
  if (n10 === 1) return "one";
  return n10 >= 2 && n10 <= 4 ? "few" : "many";
}

function tripletWords(number, c, gender) {
  const words = [], i = CASE_INDEX[c];
  if (number >= 100) words.push(CARDINAL[Math.floor(number / 100) * 100][i]);
  let rest = number % 100;
  if (rest >= 20) { words.push(CARDINAL[Math.floor(rest / 10) * 10][i]); rest %= 10; }
  if (rest === 1) words.push(ONE[gender][i]);
  else if (rest === 2) words.push(TWO[gender][i]);
  else if (rest) words.push(CARDINAL[rest][i]);
  return words;
}

function scaleWord(scale, group, c) {
  const s = SCALES[scale], cls = countClass(group), i = CASE_INDEX[c];
  if (c === "nom" || c === "acc") {
    if (cls === "one") return s.sg[i];
    return cls === "few" ? s.sg[1] : s.pl[1];
  }
  return cls === "one" ? s.sg[i] : s.pl[i];
}

/** Inflect cardinal n (BigInt|number) in case c and gender m/f/n (for 1 and 2). */
function cardinal(n, c = "nom", gender = "m") {
  n = BigInt(n);
  if (n >= 10n ** 15n) return [...n.toString()].map((d) => (d === "0" ? CARDINAL[0][0] : tripletWords(Number(d), "nom", "m")[0])).join(" ");
  if (n === 0n) return CARDINAL[0][CASE_INDEX[c]];
  const words = [];
  for (let scale = SCALES.length - 1; scale >= 0; scale--) {
    const group = Number((n / 1000n ** BigInt(scale)) % 1000n);
    if (group === 0) continue;
    words.push(...tripletWords(group, c, scale ? SCALES[scale].gender : gender));
    if (scale) words.push(scaleWord(scale, group, c));
  }
  return words.join(" ");
}

// Ordinals: stems and endings; h is hard, s has a stressed ending, t is the special third form.
const ORDINAL_STEMS = {
  0: ["нулев", "s"], 1: ["перв", "h"], 2: ["втор", "s"], 3: ["трет", "t"], 4: ["четвёрт", "h"], 5: ["пят", "h"],
  6: ["шест", "s"], 7: ["седьм", "s"], 8: ["восьм", "s"], 9: ["девят", "h"], 10: ["десят", "h"],
  11: ["одиннадцат", "h"], 12: ["двенадцат", "h"], 13: ["тринадцат", "h"], 14: ["четырнадцат", "h"],
  15: ["пятнадцат", "h"], 16: ["шестнадцат", "h"], 17: ["семнадцат", "h"], 18: ["восемнадцат", "h"],
  19: ["девятнадцат", "h"], 20: ["двадцат", "h"], 30: ["тридцат", "h"], 40: ["сороков", "s"],
  50: ["пятидесят", "h"], 60: ["шестидесят", "h"], 70: ["семидесят", "h"], 80: ["восьмидесят", "h"],
  90: ["девяност", "h"], 100: ["сот", "h"], 200: ["двухсот", "h"], 300: ["трёхсот", "h"], 400: ["четырёхсот", "h"],
  500: ["пятисот", "h"], 600: ["шестисот", "h"], 700: ["семисот", "h"], 800: ["восьмисот", "h"], 900: ["девятисот", "h"],
};
const HARD = { m: ["ый", "ого", "ому", "ый", "ым", "ом"], f: ["ая", "ой", "ой", "ую", "ой", "ой"],
               n: ["ое", "ого", "ому", "ое", "ым", "ом"], pl: ["ые", "ых", "ым", "ые", "ыми", "ых"] };
const ENDINGS = {
  h: HARD,
  s: { ...HARD, m: ["ой", "ого", "ому", "ой", "ым", "ом"] },
  t: { m: ["ий", "ьего", "ьему", "ий", "ьим", "ьем"], f: ["ья", "ьей", "ьей", "ью", "ьей", "ьей"],
       n: ["ье", "ьего", "ьему", "ье", "ьим", "ьем"], pl: ["ьи", "ьих", "ьим", "ьи", "ьими", "ьих"] },
};

function ordinalEnding(stem, type, c, gender) { return stem + ENDINGS[type][gender][CASE_INDEX[c]]; }

/** Build the combining stem for compound thousandth ordinals. */
function compoundPrefix(group) {
  const parts = [];
  if (group >= 100) parts.push(Math.floor(group / 100) === 1 ? "сто" : CARDINAL[Math.floor(group / 100) * 100][1]);
  let rest = group % 100;
  if (rest >= 20) { parts.push(rest - (rest % 10) === 90 ? "девяносто" : CARDINAL[rest - (rest % 10)][1]); rest %= 10; }
  if (rest === 1) parts.push(group === 1 ? "" : "одно");
  else if (rest === 2) parts.push("двух");
  else if (rest) parts.push(CARDINAL[rest][1]);
  return parts.join("");
}

/** Inflect ordinal n in case c and gender m/f/n/pl; only the last word changes. */
function ordinal(n, c = "nom", gender = "m") {
  n = BigInt(n);
  if (n === 0n) return ordinalEnding("нулев", "s", c, gender);
  const low = Number(n % 1000n);
  if (low !== 0) {
    const head = n - BigInt(low) > 0n ? ordinalHead(n - BigInt(low)) + " " : "";
    const words = [];
    let rest = low, last;
    if (rest % 100 === 0) last = rest;
    else {
      if (rest >= 100) { words.push(CARDINAL[Math.floor(rest / 100) * 100][0]); rest %= 100; }
      if (rest >= 20 && rest % 10 !== 0) { words.push(CARDINAL[rest - (rest % 10)][0]); rest %= 10; }
      last = rest;
    }
    const [stem, type] = ORDINAL_STEMS[last];
    words.push(ordinalEnding(stem, type, c, gender));
    return head + words.join(" ");
  }
  let scale = 1;
  while (n % 1000n ** BigInt(scale + 1) === 0n && scale < SCALES.length - 1) scale++;
  const group = Number((n / 1000n ** BigInt(scale)) % 1000n);
  const rest = n - BigInt(group) * 1000n ** BigInt(scale);
  const head = rest > 0n ? ordinalHead(rest) + " " : "";
  return head + ordinalEnding(compoundPrefix(group) + SCALES[scale].ordinal, "h", c, gender);
}

/** Uninflected ordinal prefix; omit the explicit feminine one before thousand. */
function ordinalHead(n) {
  return cardinal(n, "nom").replace(/^одна тысяча/, "тысяча");
}

// Context: prepositions before numbers and neighboring word endings.

const PREPOSITIONS = ["начиная с", "вплоть до", "более", "менее", "свыше", "около", "после", "кроме", "среди", "вокруг", "возле", "вдоль",
  "против", "ради", "между", "перед", "через", "обо", "для", "без", "над", "под", "про", "при", "во", "на", "об",
  "ко", "по", "со", "до", "от", "из", "за", "в", "о", "к", "с", "у"];
// Prepositions are case-insensitive even in rules preserving uppercase Roman numerals.
const anyCase = (word) => [...word].map((c) => (c === " " ? "\\s+" : `[${c}${c.toUpperCase()}]`)).join("");
const PREP_RE = `(?<prepw>${NOT_LETTER_BEFORE}(?<prep>${PREPOSITIONS.map(anyCase).join("|")})\\s+)?`;
const GEN_PREPS = new Set(["до", "от", "из", "у", "без", "для", "более", "менее", "свыше", "около", "после", "кроме", "среди", "вокруг", "возле",
  "вдоль", "против", "ради", "начиная с", "вплоть до"]);

function prepKey(prep) { return prep ? prep.toLowerCase().replace(/\s+/g, " ") : null; }

/** Case implied by a preposition alone; null means absent or ambiguous. */
function caseByPrep(prep) {
  const p = prepKey(prep);
  if (!p) return null;
  if (GEN_PREPS.has(p)) return "gen";
  if (p === "к" || p === "ко") return "dat";
  if (p === "о" || p === "об" || p === "обо" || p === "при") return "prep";
  if (p === "между" || p === "над" || p === "под" || p === "перед") return "ins";
  if (p === "через" || p === "про") return "acc";
  return null;
}

const ENDS = (word, ...endings) => endings.some((e) => word.endsWith(e));
const plain = (word) => (word || "").replace(/\+/g, "").toLowerCase();

/** Cardinal case from a preposition and the following word; default is nominative. */
function cardinalCase(prep, next) {
  const p = prepKey(prep), w = plain(next), fixed = caseByPrep(prep);
  if (fixed) return fixed === "acc" ? "nom" : fixed;
  if (p === "с" || p === "со") return ENDS(w, "ами", "ями", "ом", "ем", "ём", "ой", "ей", "ью") ? "ins" : "gen";
  if (p === "за") return ENDS(w, "ами", "ями", "ом", "ем", "ём") ? "ins" : "nom";
  if (p === "в" || p === "во" || p === "на") return ENDS(w, "ах", "ях") ? "prep" : "nom";
  return "nom";
}

/** Heuristic gender for 1 and 2 from the noun ending in case c. */
function genderByNoun(next, c, count) {
  const w = plain(next);
  if (!w) return "m";
  if (c === "nom" || c === "acc") {
    if (count === "one") {
      if (ENDS(w, "а", "я") || (c === "acc" && ENDS(w, "у", "ю"))) return "f";
      return ENDS(w, "о", "е", "ё") ? "n" : "m";
    }
    return ENDS(w, "ы", "и") ? "f" : "m";
  }
  if (count !== "one") return "m";
  if (c === "gen") return ENDS(w, "ы", "и") ? "f" : "m";
  if (c === "dat") return ENDS(w, "е", "и") ? "f" : "m";
  if (c === "ins") return ENDS(w, "ой", "ей", "ью") ? "f" : "m";
  return ENDS(w, "и") ? "f" : "m";
}

// Contextual expansion rules.

const MONTHS = plusy("января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря");
const YEAR_NOUNS = { "год": ["nom", false], "года": ["gen", false], "году": ["prep", false], "годом": ["ins", false],
  "годы": ["nom", true], "годов": ["gen", true], "годам": ["dat", true], "годами": ["ins", true], "годах": ["prep", true] };
const YEAR_ABBR = { prep: ["году", "годах"], dat: ["году", "годам"], gen: ["года", "годов"], acc: ["год", "годы"],
                    nom: ["год", "годы"], ins: ["годом", "годами"] };

function yearRule(text) {
  const re = new RegExp(`${PREP_RE}(?<a>\\d{1,4})(?:-[${L}]{1,3})?(?:\\s*[–—-]\\s*(?<b>\\d{1,4})(?:-[${L}]{1,3})?)?` +
    `\\s+(?<noun>(?:${plusy("годами|годах|годам|годов|годом|годы|году|года|год")})${NOT_LETTER_AFTER}|гг\\.|г\\.)`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1), offset = args.at(-3);
    const noun = plain(g.noun), prep = prepKey(g.prep), abbr = noun.endsWith(".");
    if (abbr && (g.a.length < 3 || (g.b && g.b.length < 3))) return whole; // Small quantities with this abbreviation are more likely grams than years.
    let c, plural;
    if (abbr) {
      plural = noun === "гг.";
      if (["в", "во", "о", "об", "обо", "при"].includes(prep)) c = "prep";
      else if (["к", "ко", "по"].includes(prep)) c = "dat";
      else if (GEN_PREPS.has(prep) || prep === "с" || prep === "со") c = "gen";
      else if (["на", "за", "через", "про"].includes(prep)) c = "acc";
      else if (new RegExp(`(?:${MONTHS})\\s*$`, "i").test(text.slice(0, offset))) c = "gen";
      else c = "nom";
    } else {
      [c, plural] = YEAR_NOUNS[noun];
      if (noun === "году" && ["к", "ко", "по"].includes(prep)) c = "dat";
    }
    const nounWord = abbr ? YEAR_ABBR[c][plural ? 1 : 0] : g.noun;
    const gender = plural && !g.b ? "pl" : "m";
    const words = ordinal(g.a, c, gender) + (g.b ? " — " + ordinal(g.b, c, gender) : "");
    return `${g.prepw ?? ""}${words} ${nounWord}`;
  });
}

/** Year ranges: first year uses genitive; second agrees with the noun. */
function yearRangeRule(text) {
  const re = new RegExp(`${NOT_LETTER_BEFORE}(?<p1>с|со|от)\\s+(?<a>\\d{3,4})\\s+(?<p2>по|до)\\s+(?<b>\\d{3,4})` +
    `\\s+(?<noun>(?:${plusy("годы|годов|года|год")})${NOT_LETTER_AFTER}|гг\\.|г\\.)`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1), noun = plain(g.noun), plural = noun === "гг." || noun === "годы" || noun === "годов";
    const c = ["года", "годов"].includes(noun) || prepKey(g.p2) === "до" ? "gen" : "nom";
    const nounWord = noun.endsWith(".") ? YEAR_ABBR[c][plural ? 1 : 0] : g.noun;
    return `${g.p1} ${ordinal(g.a, "gen", "m")} ${g.p2} ${ordinal(g.b, c, "m")} ${nounWord}`;
  });
}

/** Date ranges: first day uses genitive; second follows the range preposition. */
function dateRangeRule(text) {
  const re = new RegExp(`${NOT_LETTER_BEFORE}(?<p1>с|со|от)\\s+(?<a>\\d{1,2})\\s+(?<p2>по|до)\\s+(?<b>\\d{1,2})` +
    `\\s+(?<month>${MONTHS})${NOT_LETTER_AFTER}`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1);
    if (Number(g.a) > 31 || Number(g.b) > 31) return whole;
    const c = prepKey(g.p2) === "по" ? "acc" : "gen";
    return `${g.p1} ${ordinal(g.a, "gen", "n")} ${g.p2} ${ordinal(g.b, c, "n")} ${g.month}`;
  });
}

function dateRule(text) {
  const re = new RegExp(`${PREP_RE}(?<d>\\d{1,2})(?:-?(?:го|е|ое|ого|му))?\\s+(?<month>${MONTHS})${NOT_LETTER_AFTER}`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1), day = Number(g.d), prep = prepKey(g.prep);
    if (day < 1 || day > 31) return whole;
    let c = "gen";
    if (prep === "к" || prep === "ко") c = "dat";
    else if (["на", "по", "через", "про", "за"].includes(prep)) c = "acc";
    return `${g.prepw ?? ""}${ordinal(day, c, "n")} ${g.month}`;
  });
}

const DECADE_CASES = { "е": "nom", "х": "prep", "м": "dat", "ми": "ins" };

function decadeRule(text) {
  // Decade suffixes; small day numbers are handled by suffixRule instead.
  const re = new RegExp(`(?<![\\d])(?<n>\\d{2,3}0|[2-9]0)-(?<suf>ми|е|х|м)${NOT_LETTER_AFTER}`, "g");
  return text.replace(re, (whole, n, suf) => ordinal(n, DECADE_CASES[suf], "pl"));
}

const SUFFIXES = [
  [["ыми", "ими", "ми"], "pl", "ins"], [["ого", "его", "го"], "m", "gen"], [["ому", "ему", "му"], "m", "dat"],
  [["ый", "ий", "ой", "й"], "m", "nom"], [["ая", "яя", "ья", "я"], "f", "nom"], [["ую", "юю", "ю"], "f", "acc"],
  [["ое", "ее", "ье", "е"], "n", "nom"], [["ые", "ие", "ьи"], "pl", "nom"], [["ых", "их", "х"], "pl", "prep"],
  [["ым", "им"], "m", "ins"], [["ом", "ем"], "m", "prep"], [["м"], "m", null],
];

function suffixRule(text) {
  const all = SUFFIXES.flatMap(([list]) => list).sort((a, b) => b.length - a.length).join("|");
  const re = new RegExp(`${PREP_RE}(?<n>\\d+)-(?<suf>${all})${NOT_LETTER_AFTER}(?<after>\\s+(?<next>[${L}+]+))?`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1), suf = g.suf.toLowerCase(), prep = prepKey(g.prep), next = plain(g.next);
    let [, gender, c] = SUFFIXES.find(([list]) => list.includes(suf));
    if (["й", "ой", "ый", "ий"].includes(suf) && ENDS(next, "е", "и", "ы", "ой", "ей", "ью")) {
      // Feminine ordinal suffixes with oblique case prepositions.
      gender = "f";
      const fixed = caseByPrep(prep);
      if (fixed && fixed !== "acc") c = fixed;
      else if (ENDS(next, "ой", "ей", "ью")) c = "ins";
      else if (ENDS(next, "ы")) c = "gen";
      else c = ["к", "ко", "по"].includes(prep) ? "dat" : "prep";
    }
    if (c === null) c = ["в", "во", "на", "о", "об", "обо", "при"].includes(prep) ? "prep" : "ins";
    return `${g.prepw ?? ""}${ordinal(g.n, c, gender)}${g.after ?? ""}`;
  });
}

// Roman numerals.
const ROMAN = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };

function romanValue(s) {
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = ROMAN[s[i]], next = ROMAN[s[i + 1]] ?? 0;
    total += v < next ? -v : v;
  }
  return total > 0 && total < 4000 && toRoman(total) === s ? total : null;
}

function toRoman(n) {
  const table = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"],
    [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
  let out = "";
  for (const [v, r] of table) while (n >= v) { out += r; n -= v; }
  return out;
}

// Numbered nouns: gender and case from their endings.
const COUNTED_NOUNS = [
  ["век", "m"], ["съезд", "m"], ["созыв", "m"], ["том", "m"], ["класс", "m"], ["раздел", "m"], ["конгресс", "m"],
  ["акт", "m"], ["сезон", "m"], ["чемпионат", "m"], ["турнир", "m"], ["этап", "m"], ["курс", "m"], ["пункт", "m"],
  ["глав", "f"], ["част", "f"], ["олимпиад", "f"], ["книг", "f"], ["сцен", "f"], ["сери", "f"], ["картин", "f"],
  ["столети", "n"], ["тысячелети", "n"], ["действи", "n"], ["явлени", "n"],
];

/** Read gender and case from COUNTED_NOUNS; return null for unknown nouns. */
function countedNoun(word, prep) {
  const w = plain(word);
  for (const [stem, gender] of COUNTED_NOUNS) {
    if (!w.startsWith(stem)) continue;
    const end = w.slice(stem.length), p = prepKey(prep);
    const dativeOrPrep = ["к", "ко", "по"].includes(p) ? "dat" : "prep";
    let c = null;
    if (gender === "m") c = { "": "nom", "а": "gen", "у": "dat", "ом": "ins", "е": "prep", "ов": "gen", "ах": "prep", "ам": "dat", "ами": "ins", "и": "nom", "ы": "nom" }[end];
    if (gender === "f") c = { "а": "nom", "ь": "nom", "я": "nom", "ы": "gen", "и": ["в", "во", "на", "о", "об", "при"].includes(p) ? "prep" : ["к", "ко", "по"].includes(p) ? "dat" : "gen", "е": dativeOrPrep, "у": "acc", "ю": "acc", "ой": "ins", "ью": "ins", "ей": "ins" }[end];
    if (gender === "n") c = { "е": "nom", "я": "gen", "ю": "dat", "ем": "ins", "и": "prep" }[end];
    if (c === undefined || c === null) return null;
    return { gender, c };
  }
  return null;
}

function romanAfterNounRule(text) {
  // Number after a noun, such as chapter IV or volume II.
  const re = new RegExp(`${PREP_RE}(?<noun>(?:[Гг]лав|[Чч]аст|[Кк]ниг|[Сс]цен|[Рр]аздел|[Дд]ействи|[Яя]влени|[Кк]артин|Том|Акт)[${L}+]*)` +
    `\\s+(?<num>[IVXLCDM]+|\\d{1,3})(?![A-Za-z0-9-])`, "g");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1), info = countedNoun(g.noun, g.prep);
    const value = /^\d+$/.test(g.num) ? Number(g.num) : romanValue(g.num);
    if (!info || !value) return whole;
    return `${g.prepw ?? ""}${g.noun} ${ordinal(value, info.c, info.gender)}`;
  });
}

const CENTURY_ABBR = { nom: ["век", "века"], gen: ["века", "веков"], dat: ["веку", "векам"], acc: ["век", "века"],
                       ins: ["веком", "веками"], prep: ["веке", "веках"] };

function romanBeforeNounRule(text) {
  // Centuries, congresses, and ranges of Roman numerals.
  const re = new RegExp(`${PREP_RE}(?<![A-Za-z])(?<a>[IVXLCDM]+)(?:\\s*[–—-]\\s*(?<b>[IVXLCDM]+))?\\s+(?<noun>вв\\.|в\\.|[${L}+]+)`, "g");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1);
    if (g.noun === "в." || g.noun === "вв.") {
      const a = romanValue(g.a), b = g.b ? romanValue(g.b) : null, plural = g.noun === "вв.";
      if (!a || (g.b && !b)) return whole;
      const p = prepKey(g.prep);
      let c = caseByPrep(g.prep) ?? (["в", "во", "на"].includes(p) ? "prep" : ["с", "со"].includes(p) ? "gen" : "nom");
      if (c === "acc") c = "nom";
      const words = ordinal(a, c, "m") + (b ? " — " + ordinal(b, c, "m") : "");
      return `${g.prepw ?? ""}${words} ${CENTURY_ABBR[c][plural ? 1 : 0]}`;
    }
    const info = countedNoun(g.noun, g.prep);
    const a = romanValue(g.a), b = g.b ? romanValue(g.b) : null;
    if (!info || !a || (g.b && !b)) return whole;
    const words = ordinal(a, info.c, info.gender) + (b ? " — " + ordinal(b, info.c, info.gender) : "");
    return `${g.prepw ?? ""}${words} ${g.noun}`;
  });
}

const FEMALE_NAMES = ["Екатерин", "Елизавет", "Анн", "Мари", "Изабелл", "Виктори", "Ирин", "Ольг", "Софь", "Софи",
  "Маргарит", "Елен", "Кристин", "Анастаси", "Наталь", "Евдоки"];

function monarchRule(text) {
  // Regnal numbers following personal names, with case agreement.
  const re = new RegExp(`(?<name>[А-ЯЁ][а-яё]+)\\s+(?<r>[IVX]{1,6})(?![A-Za-z0-9])`, "g");
  return text.replace(re, (whole, name, r) => {
    const value = romanValue(r);
    if (!value) return whole;
    const female = FEMALE_NAMES.some((stem) => name.startsWith(stem));
    let c;
    if (female) c = ENDS(name, "а", "я") ? "nom" : ENDS(name, "ы", "и") ? "gen" : ENDS(name, "у", "ю") ? "acc" : ENDS(name, "ой", "ей") ? "ins" : "prep";
    else c = ENDS(name, "ом", "ем", "ём") ? "ins" : ENDS(name, "а", "я") ? "gen" : ENDS(name, "у", "ю") ? "dat" : ENDS(name, "е") ? "prep" : "nom";
    const words = ordinal(value, c, female ? "f" : "m");
    return `${name} ${words[0].toUpperCase()}${words.slice(1)}`;
  });
}

const PERCENT = { one: ["процент", "процента", "проценту", "процент", "процентом", "проценте"],
                  few: ["процента", "процентов", "процентам", "процента", "процентами", "процентах"],
                  many: ["процентов", "процентов", "процентам", "процентов", "процентами", "процентах"] };

function percentRule(text) {
  const re = new RegExp(`${PREP_RE}(?<n>\\d{1,3}(?:[ \\u00a0\\u202f]\\d{3})+|\\d+)(?<frac>[.,]\\d{1,3})?\\s?%`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1);
    if (g.frac) return `${g.prepw ?? ""}${g.n}${g.frac} процента`; // text.js expands fractions; use genitive singular for the percentage noun.
    let c = caseByPrep(g.prep) ?? (["с", "со"].includes(prepKey(g.prep)) ? "gen" : "nom");
    if (c === "acc") c = "nom";
    const n = BigInt(g.n.replace(/\D/g, "")), cls = countClass(n);
    return `${g.prepw ?? ""}${cardinal(n, c, "m")} ${PERCENT[cls][CASE_INDEX[c]]}`;
  });
}

function timeRule(text) {
  const re = new RegExp(`${PREP_RE}(?<![\\d:])(?<h>[01]?\\d|2[0-3]):(?<m>[0-5]\\d)(?![\\d:])`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1);
    let c = caseByPrep(g.prep) ?? "nom";
    if (c === "acc") c = "nom";
    const minutes = g.m === "00" ? "ноль ноль" : (g.m[0] === "0" ? "ноль " : "") + cardinal(Number(g.m), c, "f");
    return `${g.prepw ?? ""}${cardinal(Number(g.h), c, "m")} ${minutes}`;
  });
}

// Frequent feminine nouns with ordinal numbering; unknown nouns default to masculine.
const FEMALE_NOUNS = ["глав", "част", "страниц", "строк", "строчк", "лини", "улиц", "квартир", "палат", "комнат",
  "сери", "книг", "сцен", "картин", "недел", "задач", "лекци", "полк", "позици", "школ", "групп", "команд",
  "очеред", "аудитори", "колонк", "граф", "таблиц", "клетк", "палуб", "платформ", "секци", "ступен"];

/** Remaining integers: case from context, gender for 1 and 2 from the noun. */
function cardinalRule(text) {
  const re = new RegExp(`${PREP_RE}(?<n>\\d{1,3}(?:[ \\u00a0\\u202f]\\d{3})+(?!\\d)|\\d+)(?<after>\\s+(?<next>[${L}+]+))?`, "gi");
  return text.replace(re, (whole, ...args) => {
    const g = args.at(-1), n = BigInt(g.n.replace(/\D/g, ""));
    const cls = countClass(n), next = plain(g.next), p = prepKey(g.prep);
    // Singular prepositional nouns imply an ordinal number rather than a quantity.
    if (["в", "во", "на", "о", "об", "обо", "при"].includes(p) && /[^аеиоуыэюяё]е$/.test(next) && n < 10000n) {
      const female = FEMALE_NOUNS.some((stem) => next.startsWith(stem));
      return `${g.prepw}${ordinal(n, "prep", female ? "f" : "m")}${g.after}`;
    }
    let c = cardinalCase(g.prep, g.next);
    if (c === "nom" && cls === "one" && ENDS(plain(g.next), "у", "ю")) c = "acc"; // Feminine accusative one before a duration noun.
    const last = Number(n % 100n);
    const needsGender = last % 10 === 1 && last !== 11 || (last % 10 === 2 && last !== 12 && (c === "nom" || c === "acc"));
    const gender = needsGender ? genderByNoun(g.next, c, cls) : "m";
    return `${g.prepw ?? ""}${cardinal(n, c, gender)}${g.after ?? ""}`;
  });
}

/** Apply before generic expansion; Roman numerals must precede Latin transliteration. */
function applyNumberRules(text) {
  text = romanAfterNounRule(text);
  text = romanBeforeNounRule(text);
  text = monarchRule(text);
  text = yearRangeRule(text);
  text = yearRule(text);
  text = dateRangeRule(text);
  text = dateRule(text);
  text = decadeRule(text);
  text = suffixRule(text);
  text = timeRule(text);
  text = percentRule(text);
  return text;
}

globalThis.SileroNumbers = { cardinal, ordinal, applyNumberRules, cardinalRule, countClass };
