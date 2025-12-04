"""
Configuration du projet Bug Predictor - Version finale avec règles
"""

import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
BEST_MODEL_DIR = BASE_DIR / "best_model"

# Création des dossiers
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, BEST_MODEL_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configuration des datasets
DATASETS = {
    "soft_def_1": DATA_DIR / "soft_def_1.csv",
    "soft_def_JM1": DATA_DIR / "soft_def_JM1.csv",
    "soft_def_2": DATA_DIR / "soft_def_2.csv",
}

# Règles de complexité (votre spécification)
COMPLEXITY_RULES = {
    'n': {'threshold': 300, 'operator': '<'},      # n < 300
    'v': {'threshold': 1000, 'operator': '<'},     # v < 1000
    'd': {'threshold': 50, 'operator': '<'},       # d < 50
    'e': {'threshold': 500000, 'operator': '<'},   # e < 500000
    't': {'threshold': 5000, 'operator': '<'}      # t < 5000
}

# Colonnes à supprimer (selon votre spécification)
COLUMNS_TO_DROP = ['ev(g)', 'iv(g)', 'b', 'branchCount']

# Paramètres de prétraitement
PREPROCESSING = {
    "test_size": 0.2,
    "random_state": 42,
    "smote_k_neighbors": 5,
    "pca_variance_ratio": 0.95,
    "correlation_threshold": 0.85,
    "feature_selection_k": 25,
    "variance_threshold": 0.01,
}

# Hyperparamètres optimisés
"""
Configuration des modèles
"""

MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'random_state': 42,
        'eval_metric': 'logloss'
    },
    'neural_network': {
        'layer_sizes': [64, 32],
        'learning_rate': 0.001,
        'activation': 'relu',
        'dropout_rate': 0.2,
        'random_state': 42,
        'epochs': 100,
        'batch_size': 32
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42,
        'probability': True,
        'class_weight': 'balanced'
    }
}

# Métriques d'évaluation
EVALUATION_METRICS = {
    "primary": "f1_score",
    "secondary": ["roc_auc", "recall", "precision", "accuracy"],
    "thresholds": {
        "f1_score": 0.6,
        "roc_auc": 0.7,
        "recall": 0.7,
        "precision": 0.6,
    }
}

# Configuration de l'application
APP_CONFIG = {
    "streamlit_port": 8501,
    "api_port": 8000,
    "host": "0.0.0.0",
    "debug": True,
}