import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kredivo_project.preprocessing import normalize_text, tokenize_id, is_generic_praise, split_sentences

def test_normalization_maps_slang_and_keeps_negation():
    text = normalize_text("GAK bisa login, apk error bgt!!!")
    assert "tidak" in text
    assert "bisa" in text
    assert "aplikasi" in text
    assert "error" in text
    assert "banget" in text

def test_tokenizer_keeps_negation():
    tokens = tokenize_id("Saya tidak bisa login", remove_stopwords=True)
    assert "tidak" in tokens
    assert "bisa" in tokens
    assert "login" in tokens

def test_generic_praise_only_exact_match():
    assert is_generic_praise(normalize_text("Sangat membantu sekali"))
    assert not is_generic_praise(normalize_text("Sangat membantu, tapi limit tidak bisa dipakai"))

def test_sentence_splitter():
    assert split_sentences("Login gagal. Saldo tidak muncul!") == ["Login gagal.", "Saldo tidak muncul!"]


def test_common_emoji_are_preserved_as_tokens():
    text = normalize_text("Aplikasinya bagus 👍😊")
    assert "emoji_jempol" in text
    assert "emoji_senang" in text
    assert is_generic_praise(normalize_text("bagus 😊"))
    assert not is_generic_praise(normalize_text("login gagal 😡"))
