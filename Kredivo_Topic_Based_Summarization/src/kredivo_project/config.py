from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {
    "dataset_path": "data/kredivo_reviews_50k.csv",
    "text_column": "content",
    "id_column": "reviewId",
    "validation_size": 0.10,
    "test_size": 0.10,
    "random_state": 42,
    "max_fit_docs": 12000,
    "transform_batch_size": 512,
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "cpu",
    "min_topic_size": 30,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "n_keywords": 12,
    "summary_sentences": 5,
    "mmr_lambda": 0.72,
    "textrank_neighbors": 15,
    "reference_topics": 30,
    "reference_examples_per_topic": 15,
    "audit_sample_size": 200,
    "audit_min_agreement": 0.80,
    "audit_min_class_agreement": 0.70,
    "audit_min_class_support": 10,
    "stopword_removal": True,
    "save_topic_model": True
}

def load_config(path: Path | None = None) -> dict:
    path = path or CONFIG_PATH
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
        return DEFAULT_CONFIG.copy()
    user_cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_cfg)
    return cfg

def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
