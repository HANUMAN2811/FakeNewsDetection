"""
app.py
------
Flask application entry-point for Fake News Detection.

Routes
------
GET  /               → Web UI (index.html)
POST /api/predict    → JSON: {prediction, label, confidence}
GET  /api/health     → JSON: {status, model_loaded}
"""

import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    SECRET_KEY, MODEL_PATH, VECTORIZER_PATH, MIN_TEXT_LENGTH,
)
from src.utils.helpers import load_model, predict, validate_input

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# Load model once at startup
try:
    _model, _vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)
    logger.info("✅ Model ready")
except FileNotFoundError as e:
    logger.warning("⚠️  Model not loaded at startup: %s", e)
    _model, _vectorizer = None, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the Web UI."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict whether the submitted news text is REAL or FAKE.

    Request body (JSON):
        { "text": "<article text>" }

    Response (JSON):
        { "prediction": "FAKE"|"REAL", "label": 0|1, "confidence": 0.0-1.0 }
    """
    if _model is None or _vectorizer is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    text = data.get("text", "")
    is_valid, err_msg = validate_input(text, MIN_TEXT_LENGTH)
    if not is_valid:
        return jsonify({"error": err_msg}), 422

    try:
        result = predict(text, _model, _vectorizer)
        logger.info("Prediction: %s (conf=%.4f)", result["prediction"], result["confidence"])
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": "Internal server error during prediction."}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Simple health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
    }), 200


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
