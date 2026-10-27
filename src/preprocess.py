"""Text normalisation and script-based language routing.

Both training and the app import from here so the model never sees text at
serving time that was cleaned differently from training time.
"""

import html
import re

from src.config import SCRIPT_RANGES

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"^(rt\b[:\s]*)+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
# Davidson's tweets arrive with literal &#8220; style entities still in them.
NUMERIC_ENTITY_RE = re.compile(r"&#\d+;")


def clean_text(text: str) -> str:
    """Normalise a social media post without stripping the signal.

    I deliberately keep emoji, casing and punctuation. Lowercasing and emoji
    removal both cost recall on the toxic class, and the transformer tokenizer
    handles casing better than I can by hand.
    """
    if text is None:
        return ""
    t = str(text)
    t = html.unescape(t)
    t = NUMERIC_ENTITY_RE.sub(" ", t)
    t = RT_RE.sub("", t)
    t = URL_RE.sub(" <url> ", t)
    t = MENTION_RE.sub(" <user> ", t)
    # A hashtag usually carries the topic, so I keep the word and drop the hash.
    t = HASHTAG_RE.sub(r"\1", t)
    t = WS_RE.sub(" ", t)
    return t.strip()


def detect_language(text: str) -> str:
    """Route text to a language by counting script characters.

    The three Indic languages here use mutually exclusive Unicode blocks, so
    counting beats any statistical detector and costs nothing. Anything that is
    not predominantly one of those blocks falls back to English, which also
    catches romanised Hindi.
    """
    if not text:
        return "en"
    counts = {lang: 0 for lang in SCRIPT_RANGES}
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        for lang, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[lang] += 1
                break
    if letters == 0:
        return "en"
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    # Needs a real share of the string, not one stray character.
    return lang if n / letters > 0.2 else "en"
