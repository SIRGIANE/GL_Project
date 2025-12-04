"""
Test script pour prédire les bugs avec le MEILLEUR modèle entraîné.
Extraction identique au script original dans le MÊME ORDRE.
Usage: python test.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

import sys
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.utils.preprocessing import helper_for_complexity_evaluation
    HAS_HELPER = True
except Exception:
    HAS_HELPER = False


def extract_features(code_str):
    """Extrait 18 features EXACTEMENT comme dans le script original"""
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
    
    return [f1, f2, f5, f6, f7, f8, f9, f10, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]


# Find latest output
latest = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)[0]
model_path = latest / "models" / "best_model.joblib"
scaler_path = latest / "models" / "scaler.joblib"
pca_path = latest / "models" / "pca.joblib"

print(f"[INFO] Loading from: {latest.name}")

# Load model
model = joblib.load(model_path)
scaler = joblib.load(scaler_path) if scaler_path.exists() else None
pca = joblib.load(pca_path) if pca_path.exists() else None

print(f"[OK] Model: {type(model).__name__}")
print(f"[OK] Scaler: {type(scaler).__name__ if scaler else 'None'}")
print(f"[OK] PCA: {type(pca).__name__ if pca else 'None'}")

# Test codes
CODE_CLEAN = "def add(a, b):\n    return a + b"

CODE_COMPLEX = """def generate_massive():
    var_0 = var_1 = var_2 = var_3 = var_4 = var_5 = var_6 = var_7 = var_8 = var_9 = 0
    result = var_0 + var_1 + var_2 + var_3 + var_4 + var_5 + var_6 + var_7 + var_8 + var_9
    for i in range(100):
        for j in range(100):
            for k in range(50):
                result *= (var_0 * var_1 * var_2 * var_3 * var_4)
    return result"""

for name, code in [("CLEAN", CODE_CLEAN), ("COMPLEX", CODE_COMPLEX)]:
    features = extract_features(code)
    X = np.array([features], dtype=float)
    
    if scaler:
        X = scaler.transform(X)
    if pca:
        X = pca.transform(X)
    
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
    else:
        prob = float(pred)
    
    print(f"\n{name}: Pred={'🐛 BUGGY' if pred == 1 else '✅ PROPRE'} ({prob:.1%})")