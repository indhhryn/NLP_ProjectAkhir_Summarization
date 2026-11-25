from __future__ import annotations
import html
import re
import unicodedata
from collections import Counter

# Normalisasi dibuat konservatif. Negasi seperti "tidak", "belum", dan "bukan"
# sengaja dipertahankan karena dapat membalik makna sebuah ulasan.
SLANG_MAP = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak",
    "enggak": "tidak", "tdk": "tidak", "ngga": "tidak", "ndak": "tidak",
    "bgt": "banget", "bgtt": "banget", "yg": "yang", "dgn": "dengan",
    "krn": "karena", "karna": "karena", "aja": "saja", "aj": "saja",
    "apk": "aplikasi", "app": "aplikasi", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "belom": "belum", "dr": "dari", "trs": "terus",
    "tp": "tapi", "klo": "kalau", "kl": "kalau", "sm": "sama",
    "sy": "saya", "aq": "saya", "km": "kamu", "bs": "bisa",
    "gmn": "bagaimana", "knp": "kenapa", "aja": "saja"
}

# Stopword ringan: tidak menghapus negasi, kata keluhan, atau kata domain.
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "saya", "aku", "kami",
    "kita", "kamu", "anda", "dia", "mereka", "nya", "untuk", "pada", "dengan",
    "karena", "agar", "atau", "juga", "sudah", "telah", "akan", "masih", "saja",
    "aja", "dalam", "oleh", "sebagai", "ada", "adalah", "jadi", "dapat", "nya", "the", "a", "an", "is", "are", "was", "were"
}
NEGATION_AND_ISSUE_WORDS = {"tidak", "belum", "bukan", "jangan", "kurang", "gagal", "error", "macet", "bisa"}

# Hanya frasa pujian umum yang berdiri sendiri yang dikeluarkan dari kandidat topik.
# Keluhan singkat seperti "tidak bisa login" tetap dipertahankan.
# Preserve common review emoji as interpretable tokens instead of deleting every emoji.
# The untouched original review text is still retained for extractive sentence selection.
EMOJI_MAP = {
    "❤️": "emoji_suka", "❤": "emoji_suka", "😍": "emoji_suka", "🥰": "emoji_suka",
    "😀": "emoji_senang", "😃": "emoji_senang", "😄": "emoji_senang", "😊": "emoji_senang",
    "👍": "emoji_jempol", "🙏": "emoji_terima_kasih", "😂": "emoji_tertawa", "🤣": "emoji_tertawa",
    "😢": "emoji_sedih", "😭": "emoji_sedih", "😞": "emoji_kecewa", "😔": "emoji_kecewa",
    "😡": "emoji_marah", "🤬": "emoji_marah", "👎": "emoji_tidak_suka",
}

GENERIC_PRAISE = {
    "mantap", "mantap sekali", "mantap banget", "bagus", "bagus sekali", "sangat bagus",
    "baik", "baik sekali", "oke", "ok", "good", "great", "terbaik", "keren", "mantul",
    "sip", "jos", "top", "memuaskan", "sangat memuaskan", "sangat membantu",
    "sangat membantu sekali", "membantu sekali", "terima kasih", "terimakasih", "thanks",
    "thank you", "recommended", "love", "bintang lima", "5 bintang"
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_TOKEN_RE = re.compile(r"(?u)\b[\w]+\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    for symbol, token in sorted(EMOJI_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(symbol, f" {token} ")
    text = _URL_RE.sub(" ", text)
    text = text.casefold()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s'’-]", " ", text, flags=re.UNICODE)
    tokens = _TOKEN_RE.findall(text)
    tokens = [SLANG_MAP.get(t, t) for t in tokens]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def tokenize_id(value: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize_text(value))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ID or t in NEGATION_AND_ISSUE_WORDS]
    return tokens


def is_generic_praise(normalized: str) -> bool:
    tokens = normalized.strip().split()
    words = [token for token in tokens if not token.startswith("emoji_")]
    # Generic praise and emoji-only reaction reviews are not topic-informative.
    # An issue plus emoji (e.g. "login gagal 😡") remains in clustering.
    if not words and any(token.startswith("emoji_") for token in tokens):
        return True
    return " ".join(words) in GENERIC_PRAISE


def split_sentences(text: str) -> list[str]:
    # Memisahkan kalimat, tetapi mengembalikan teks asli agar ekstraksi tetap dapat dilacak.
    parts = [p.strip() for p in _SENTENCE_RE.split(str(text)) if p and p.strip()]
    return parts or ([str(text).strip()] if str(text).strip() else [])


def corpus_profile(texts: list[str]) -> dict:
    lengths = [len(tokenize_id(t, remove_stopwords=False)) for t in texts]
    counts = Counter(lengths)
    return {
        "n_documents": len(texts),
        "median_tokens": float(sorted(lengths)[len(lengths)//2]) if lengths else 0.0,
        "one_or_two_tokens": sum(n <= 2 for n in lengths),
        "one_sentence_share": sum(len(split_sentences(t)) <= 1 for t in texts) / max(1, len(texts)),
        "token_length_distribution": dict(counts),
    }
