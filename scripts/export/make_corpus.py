"""Build a stress parity corpus from local cp1251 HTML plus difficult sample sentences.

Usage: python scripts/export/make_corpus.py INPUT_HTML OUT_TXT [LIMIT]
Selects the first LIMIT Cyrillic sentences of 3-300 characters (default 400).
The input text must be provided separately and is not distributed with this project."""

import html
import re
import sys
from pathlib import Path

EXTRA = [
    "Старый замок на холме был виден издалека.",
    "Ключ застрял в замке, и замок пришлось сломать.",
    "Мука для пирогов кончилась, и это была настоящая мука.",
    "Все пришли вовремя, и всё прошло хорошо.",
    "Берег реки зарос камышом и ивами.",
    "Б+ерег рек+и зарос камышом и ивами.",
    "Старый з+амок у рек+и.",
    "Он сидел у окна и ждал её до самого вечера.",
    "Ёлка стояла в углу, а ежик спал под ней.",
    "Кто-то постучал, и что-то упало на пол.",
    "Где-нибудь мы обязательно встретимся, как-нибудь потом.",
    "В 2024 году было 15 событий и 3,5 процента роста.",
    "Слово hello значит привет, а Python — язык программирования.",
    "Почему ты молчишь? Ответь мне хоть что-нибудь!",
    "Ты правда придёшь завтра?",
    "Здравствуйте, так звучит этот голос.",
    "Он плачет: «Я вернусь!» — и уходит.",
    "Атлас мира лежал на столе, рядом с отрезом атласа.",
    "Органы власти и органы чувств — разные вещи.",
    "Он не мог понять, почему стрелки часов стоят.",
    "Уж замуж невтерпёж.",
    "Да.",
    "C++ и 2+2 — это не ударения.",
    "Писатель писал о писателях, а дети играли в снежки.",
    "Сорока-белобока кашу варила, деток кормила.",
]


def main() -> int:
    raw = Path(sys.argv[1]).read_bytes().decode("cp1251", errors="replace")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"\s+", " ", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text)]
    picked = [s for s in sentences if 3 <= len(s) <= 300 and re.search("[а-яё]", s, re.I)
              and "lib.ru" not in s.lower() and "©" not in s]
    lines = EXTRA + picked[:limit]
    Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(lines), "sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
