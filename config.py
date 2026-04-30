"""
config.py
---------
Centralised configuration for Fake News Detection.
All file paths, hyper-parameters, and Flask settings live here.
Import this module everywhere instead of hard-coding paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Directories ──────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR         = BASE_DIR / "models"
TEMPLATES_DIR      = BASE_DIR / "templates"
STATIC_DIR         = BASE_DIR / "static"

# Auto-create writable dirs at import time
for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset Paths ─────────────────────────────────────────────────────────────
FAKE_CSV      = RAW_DATA_DIR / "Fake.csv"
TRUE_CSV      = RAW_DATA_DIR / "True.csv"
PROCESSED_CSV = PROCESSED_DATA_DIR / "news_dataset.csv"

# ── Saved Model Artefacts ─────────────────────────────────────────────────────
MODEL_PATH      = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

# ── TF-IDF Hyper-parameters ───────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams
TFIDF_SUBLINEAR_TF = True

# ── Train / Test Split ────────────────────────────────────────────────────────
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# ── Label Encoding ────────────────────────────────────────────────────────────
FAKE_LABEL = 1   # Fake news  → 1
REAL_LABEL = 0   # Real news  → 0
LABEL_MAP  = {REAL_LABEL: "REAL", FAKE_LABEL: "FAKE"}

# ── Input Validation ─────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 20   # reject inputs shorter than this

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST  = os.getenv("FLASK_HOST",  "127.0.0.1")
FLASK_PORT  = int(os.getenv("FLASK_PORT",  "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
SECRET_KEY  = os.getenv("SECRET_KEY",  "change-me-in-production")
