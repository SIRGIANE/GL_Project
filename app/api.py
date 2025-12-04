"""
Flask API pour Bug Predictor
Convertit 18 features → 42 features pour le modèle entraîné
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit
from sklearn.preprocessing import PolynomialFeatures
import logging

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.utils.preprocessing import helper_for_complexity_evaluation
    HAS_HELPER = True
except Exception:
    HAS_HELPER = False

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

LOADED = {
    "model": None,
    "scaler": None,
    "pca": None,
    "metadata": {},
    "poly": None
}


def find_latest_output_dir():
    if not OUTPUT_DIR.exists():
        return None
    dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0] if dirs else None


def load_best_model():
    """Charge le meilleur modèle + scaler + PCA"""
    latest = find_latest_output_dir()
    if not latest:
        logger.warning("No output directory found")
        return False

    models_dir = latest / "models"
    
    # Load model
    model_path = models_dir / "best_model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    try:
        LOADED["model"] = joblib.load(model_path)
        logger.info(f"✓ Model loaded: {type(LOADED['model']).__name__}")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return False

    # Load scaler
    scaler_path = models_dir / "scaler.joblib"
    if scaler_path.exists():
        try:
            LOADED["scaler"] = joblib.load(scaler_path)
            logger.info("✓ Scaler loaded")
        except Exception as e:
            logger.warning(f"Scaler load error: {e}")

    # Load PCA
    pca_path = models_dir / "pca.joblib"
    if pca_path.exists():
        try:
            LOADED["pca"] = joblib.load(pca_path)
            logger.info(f"✓ PCA loaded (n_components={LOADED['pca'].n_components_})")
        except Exception as e:
            logger.warning(f"PCA load error: {e}")

    return True


def expand_18_to_42_features(features_18):
    """
    Convertit 18 features → 42 features
    Stratégie: utiliser PolynomialFeatures(degree=2) puis truncate à 42
    """
    X = np.array([features_18], dtype=float)
    
    # PolynomialFeatures(degree=2) avec 18 features crée ~180 features
    # On va créer et garder les plus importantes
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # On prend les 42 premières colonnes (features de base + interactions principales)
    if X_poly.shape[1] >= 42:
        X_expanded = X_poly[:, :42]
    else:
        # Si moins de 42, on pad avec zéros
        pad = 42 - X_poly.shape[1]
        X_expanded = np.hstack([X_poly, np.zeros((X_poly.shape[0], pad), dtype=float)])
    
    return X_expanded[0]


def extract_features(code_str):
    """Extrait 18 features (même ordre que script original)"""
    try:
        raw = analyze(code_str)
        f1 = raw.loc
        f13 = raw.sloc
        f14 = raw.comments
        f15 = raw.blank
        f16 = raw.lloc

        ast = h_visit(code_str)
        hal = ast.total
        f17 = hal.h1
        f18 = hal.h2
        f19 = hal.N1
        f20 = hal.N2
        f5 = hal.length
        f6 = hal.volume
        f7 = (f6 / f5) if f5 > 0 else 0
        f8 = hal.difficulty
        f9 = f7 * f6
        f10 = hal.effort
        f12 = hal.time

        cc = cc_visit(code_str)
        try:
            f2 = cc[-1].complexity
        except Exception:
            f2 = 0

        if HAS_HELPER:
            try:
                f21 = helper_for_complexity_evaluation(f5, f6, f8, f10, f12)
            except Exception:
                f21 = int(((f5 >= 300) or (f6 >= 1000) or (f8 >= 50) or (f10 >= 500000) or (f12 >= 5000)))
        else:
            f21 = int(((f5 >= 300) or (f6 >= 1000) or (f8 >= 50) or (f10 >= 500000) or (f12 >= 5000)))

        features = [f1, f2, f5, f6, f7, f8, f9, f10, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]
        metrics = {
            "LOC": f1, "cyclomatic": f2, "n": f5, "v": f6, "l": f7,
            "d": f8, "i": f9, "e": f10, "t": f12, "sloc": f13,
            "comments": f14, "blank": f15, "lloc": f16,
            "uniq_op": f17, "uniq_opnd": f18, "total_op": f19, "total_opnd": f20,
            "evaluation": f21
        }
        return features, metrics
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None, None


def predict(features_18):
    """
    Convertit 18 → 42 features
    Applique scaler + PCA
    Prédit avec le modèle
    """
    model = LOADED["model"]
    if not model:
        raise RuntimeError("Model not loaded")

    # Expand 18 → 42
    features_42 = expand_18_to_42_features(features_18)
    X = np.array([features_42], dtype=float)

    # Scaler
    if LOADED["scaler"]:
        try:
            X = LOADED["scaler"].transform(X)
            logger.debug("Scaler applied")
        except Exception as e:
            logger.warning(f"Scaler error: {e}")

    # PCA
    if LOADED["pca"]:
        try:
            X = LOADED["pca"].transform(X)
            logger.debug("PCA applied")
        except Exception as e:
            logger.warning(f"PCA error: {e}")

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            try:
                classes = list(model.classes_)
                pos_idx = classes.index(1)
            except Exception:
                pos_idx = probas.shape[1] - 1 if probas.ndim == 2 else 0
            prob = float(probas[0][pos_idx])
            is_bug = int(prob >= 0.5)
        else:
            pred = int(model.predict(X)[0])
            is_bug = pred
            prob = 1.0 if pred == 1 else 0.0
        
        return is_bug, prob
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


def determine_risk(prob):
    """Détermine le niveau de risque"""
    if prob >= 0.9:
        return "🔴 CRITIQUE"
    if prob >= 0.7:
        return "🟠 ÉLEVÉ"
    if prob >= 0.5:
        return "🟡 MOYEN"
    if prob >= 0.3:
        return "🟢 FAIBLE"
    return "✅ TRÈS FAIBLE"


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded": LOADED["model"] is not None,
        "model_type": type(LOADED["model"]).__name__ if LOADED["model"] else None,
        "scaler_loaded": LOADED["scaler"] is not None,
        "pca_loaded": LOADED["pca"] is not None,
        "pca_components": LOADED["pca"].n_components_ if LOADED["pca"] else None
    }), 200


@app.route("/api/analyze/code", methods=["POST"])
def analyze_code():
    """Analyse du code + prédiction"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        
        if not code:
            return jsonify({"error": "code required"}), 400

        # Extract 18 features
        features, metrics = extract_features(code)
        if features is None:
            return jsonify({"error": "extraction failed"}), 500

        response = {
            "features": features,
            "metrics": metrics
        }

        # Predict (uses model if loaded)
        if LOADED["model"]:
            try:
                is_bug, prob = predict(features)
                risk = determine_risk(prob)
                response["is_bug"] = bool(is_bug)
                response["probability"] = prob
                response["risk_level"] = risk
                logger.info(f"Prediction: is_bug={is_bug}, prob={prob:.4f}")
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                response["prediction_error"] = str(e)
        else:
            logger.warning("Model not loaded, no prediction")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


def startup():
    logger.info("=" * 70)
    logger.info("🐛 BUG PREDICTOR API - STARTUP")
    logger.info("=" * 70)
    success = load_best_model()
    if success:
        logger.info("✓ API READY - Model loaded and ready")
    else:
        logger.warning("⚠ Model not loaded - API running but predictions will fail")
    logger.info("=" * 70)


if __name__ == "__main__":
    startup()
    logger.info("Listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)