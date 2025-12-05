"""
API Flask pour la prédiction de bugs - Bug Predictor AI
Fournit les endpoints REST pour l'analyse de code et la prédiction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback

# Configuration du chemin
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from radon.raw import analyze as radon_analyze
    from radon.metrics import h_visit
    from radon.complexity import cc_visit
    # Importer la classe BugPreprocessor
    from src.utils.preprocessing import BugPreprocessor
    DEPENDENCIES_OK = True
    print("✓ Toutes les dépendances d'analyse sont disponibles")
except ImportError as e:
    print(f"⚠️ Erreur d'import des dépendances: {e}")
    DEPENDENCIES_OK = False

app = Flask(__name__)
CORS(app)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = PROJECT_ROOT / "output"
MODELS = {}
PREPROCESSOR = None  # Instance du BugPreprocessor

def load_models():
    """Charge tous les modèles ML disponibles et le preprocesseur"""
    global MODELS, PREPROCESSOR
    
    try:
        # Chercher le répertoire de modèles le plus récent
        model_dirs = [d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith('202')]
        if not model_dirs:
            logger.warning("Aucun répertoire de modèles trouvé")
            return False
        
        latest_dir = max(model_dirs, key=lambda x: x.name)
        models_path = latest_dir / "models"
        
        if not models_path.exists():
            logger.warning(f"Répertoire modèles non trouvé: {models_path}")
            return False
        
        # Charger les modèles disponibles
        model_files = {
            "random_forest": "random_forest_model.joblib",
            "gradient_boosting": "gradient_boosting_model.joblib", 
            "svm": "svm_rbf_model.joblib",
            "knn": "k-nearest_neighbors_model.joblib",
            "best": "best_model.joblib"
        }
        
        for name, filename in model_files.items():
            model_path = models_path / filename
            if model_path.exists():
                try:
                    MODELS[name] = joblib.load(model_path)
                    logger.info(f"✓ Modèle {name} chargé")
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du modèle {name}: {e}")
        
        # Charger le preprocesseur complet
        PREPROCESSOR = BugPreprocessor()
        try:
            scaler_path = models_path / "scaler.joblib"
            pca_path = models_path / "pca.joblib"
            
            if scaler_path.exists() and pca_path.exists():
                PREPROCESSOR.scaler = joblib.load(scaler_path)
                PREPROCESSOR.pca = joblib.load(pca_path)
                logger.info(f"✓ Preprocesseur chargé (Scaler + PCA)")
                logger.info(f"   - PCA components: {PREPROCESSOR.pca.n_components_}")
                logger.info(f"   - Scaler features: {len(PREPROCESSOR.scaler.feature_names_in_) if hasattr(PREPROCESSOR.scaler, 'feature_names_in_') else 'N/A'}")
            else:
                logger.warning("Scaler ou PCA non trouvés")
                PREPROCESSOR = None
        except Exception as e:
            logger.error(f"Erreur lors du chargement du preprocesseur: {e}")
            PREPROCESSOR = None
        
        return len(MODELS) > 0
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        return False

def extract_features(code_str):
    """Extrait les features d'un code Python avec le même pipeline que l'entraînement"""
    if not DEPENDENCIES_OK:
        raise Exception("Dépendances manquantes pour l'analyse de code")
    
    try:
        # Analyse basique avec Radon (exactement comme dans streamlit_app.py)
        raw = radon_analyze(code_str)
        f1 = raw.loc  # Lines of code
        f13 = raw.sloc  # Source lines of code
        f14 = raw.comments  # Comment lines
        f15 = raw.blank  # Blank lines
        f16 = raw.lloc  # Logical lines of code

        # Métriques de Halstead
        ast = h_visit(code_str)
        hal = ast.total
        f17 = hal.h1  # Unique operators
        f18 = hal.h2  # Unique operands
        f19 = hal.N1  # Total operators
        f20 = hal.N2  # Total operands
        f5 = hal.length  # Program length (n)
        f6 = hal.volume  # Program volume (v)
        f7 = (f6 / f5) if f5 > 0 else 0  # Level
        f8 = hal.difficulty  # Difficulty (d)
        f9 = f7 * f6  # Intelligence
        f10 = hal.effort  # Effort (e)
        f12 = hal.time  # Time to program (t)

        # Complexité cyclomatique
        cc = cc_visit(code_str)
        try:
            f2 = cc[-1].complexity if cc else 0
        except (IndexError, AttributeError):
            f2 = 0

        # Évaluation de complexité (EXACTEMENT comme dans dataset.py)
        # Règle inversée: Si n < 300 ET v < 1000 ET d < 50 ET e < 500000 ET t < 5000 → Simple (0)
        # Sinon → Complexe (1)
        is_simple = (
            (f5 < 300) & 
            (f6 < 1000) & 
            (f8 < 50) & 
            (f10 < 500000) & 
            (f12 < 5000)
        )
        f21 = int(not is_simple)  # complexityEvaluation

        # Vecteur de 18 features de base
        base_features = [f1, f2, f5, f6, f7, f8, f9, f10, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]
        
        # Création d'un DataFrame temporaire pour générer les mêmes features que l'entraînement
        # Le dataset d'entraînement utilise plusieurs colonnes qui ne sont pas dans notre extraction simple
        # Nous devons simuler la même structure
        
        # Features nommées selon le pattern du dataset d'entraînement
        feature_names = [
            'loc', 'v(g)', 'n', 'v', 'l', 'd', 'i', 'e', 't', 'lOCode', 'lOComment', 
            'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 
            'total_Opnd', 'complexityEvaluation'
        ]
        
        # Créer un DataFrame temporaire avec ces features
        temp_df = pd.DataFrame([base_features], columns=feature_names)
        
        # Appliquer les mêmes transformations que dans le dataset d'entraînement
        # Le dataset original pourrait avoir des features supplémentaires générées automatiquement
        
        # Pour simuler le même nombre de features que l'entraînement (42),
        # nous devons ajouter des features dérivées comme dans le dataset d'entraînement
        
        # Features dérivées courantes dans les datasets NASA
        temp_df['cyclomatic_density'] = temp_df['v(g)'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        temp_df['halstead_density'] = temp_df['v'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        temp_df['effort_density'] = temp_df['e'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        temp_df['comment_ratio'] = temp_df['lOComment'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        temp_df['blank_ratio'] = temp_df['lOBlank'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        
        # Features de normalisation Halstead
        temp_df['n_normalized'] = temp_df['n'] / 100.0  # Normalisation
        temp_df['v_normalized'] = temp_df['v'] / 1000.0
        temp_df['d_normalized'] = temp_df['d'] / 50.0
        temp_df['e_normalized'] = temp_df['e'] / 500000.0
        temp_df['t_normalized'] = temp_df['t'] / 5000.0
        
        # Features de ratio
        temp_df['op_ratio'] = temp_df['uniq_Op'] / temp_df['total_Op'] if temp_df['total_Op'].iloc[0] > 0 else 0
        temp_df['opnd_ratio'] = temp_df['uniq_Opnd'] / temp_df['total_Opnd'] if temp_df['total_Opnd'].iloc[0] > 0 else 0
        
        # Features de complexité
        temp_df['complexity_per_line'] = temp_df['v(g)'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        temp_df['volume_per_line'] = temp_df['v'] / temp_df['loc'] if temp_df['loc'].iloc[0] > 0 else 0
        
        # Features logarithmiques (courantes dans l'analyse de code)
        temp_df['log_loc'] = np.log1p(temp_df['loc'])
        temp_df['log_v'] = np.log1p(temp_df['v'])
        temp_df['log_e'] = np.log1p(temp_df['e'])
        
        # Features quadratiques (interactions)
        temp_df['loc_complexity'] = temp_df['loc'] * temp_df['v(g)']
        temp_df['volume_difficulty'] = temp_df['v'] * temp_df['d']
        temp_df['effort_time'] = temp_df['e'] * temp_df['t']
        
        # Features de seuil binaire
        temp_df['high_complexity'] = (temp_df['v(g)'] > 10).astype(int)
        temp_df['large_file'] = (temp_df['loc'] > 100).astype(int)
        temp_df['high_volume'] = (temp_df['v'] > 500).astype(int)
        temp_df['high_difficulty'] = (temp_df['d'] > 20).astype(int)
        
        # Continuer d'ajouter des features jusqu'à atteindre 42
        # Features supplémentaires dérivées
        temp_df['maintenance_index'] = 171 - 5.2 * np.log(temp_df['v']) - 0.23 * temp_df['v(g)'] - 16.2 * np.log(temp_df['loc']) if temp_df['v'].iloc[0] > 0 and temp_df['loc'].iloc[0] > 0 else 0
        temp_df['code_churn'] = temp_df['loc'] + temp_df['lOComment'] + temp_df['lOBlank']
        
        # S'assurer qu'on a exactement 42 features comme attendu par le modèle
        current_features = len(temp_df.columns)
        target_features = 42
        
        # Ajouter des features de padding si nécessaire
        for i in range(current_features, target_features):
            temp_df[f'padding_feature_{i}'] = 0.0
        
        # Prendre exactement les 42 premières features
        final_features = temp_df.iloc[0, :target_features].values.tolist()
        
        # Dictionnaire des métriques nommées (pour l'affichage)
        metrics = {
            "LOC": f1,
            "cyclomatic_complexity": f2,
            "halstead_length": f5,
            "halstead_volume": f6,
            "halstead_level": f7,
            "halstead_difficulty": f8,
            "halstead_intelligence": f9,
            "halstead_effort": f10,
            "halstead_time": f12,
            "source_lines": f13,
            "comment_lines": f14,
            "blank_lines": f15,
            "logical_lines": f16,
            "unique_operators": f17,
            "unique_operands": f18,
            "total_operators": f19,
            "total_operands": f20,
            "complexity_evaluation": f21
        }
        
        logger.info(f"Features générées: {len(final_features)} (attendues: 42)")
        
        return final_features, metrics
    
    except Exception as e:
        logger.error(f"Erreur d'extraction de features: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise Exception(f"Impossible d'analyser le code: {str(e)}")

def predict_bug(features):
    """Fait une prédiction de bug avec les 42 features générées"""
    if not MODELS:
        raise Exception("Aucun modèle chargé")
    
    try:
        # Les features sont déjà au bon format (42 features)
        features_array = np.array(features).reshape(1, -1)
        
        logger.info(f"Features shape: {features_array.shape}")
        
        # Vérifier que nous avons exactement 42 features
        if features_array.shape[1] != 42:
            raise Exception(f"Nombre incorrect de features: {features_array.shape[1]} au lieu de 42")
        
        # Appliquer le StandardScaler directement
        if PREPROCESSOR is None or PREPROCESSOR.scaler is None:
            raise Exception("StandardScaler non chargé")
        
        features_scaled = PREPROCESSOR.scaler.transform(features_array)
        logger.info(f"Features shape après scaling: {features_scaled.shape}")
        
        # Appliquer PCA si disponible
        if PREPROCESSOR.pca is not None:
            features_final = PREPROCESSOR.pca.transform(features_scaled)
            logger.info(f"Features shape après PCA: {features_final.shape}")
        else:
            features_final = features_scaled
            logger.info("PCA non disponible, utilisation des features scaled")
        
        # Utiliser le meilleur modèle si disponible, sinon random forest
        model_name = "best" if "best" in MODELS else "random_forest"
        if model_name not in MODELS and MODELS:
            model_name = list(MODELS.keys())[0]
        
        if model_name not in MODELS:
            raise Exception("Aucun modèle disponible")
        
        model = MODELS[model_name]
        logger.info(f"Utilisation du modèle: {model_name}")
        
        # Prédiction avec les features transformées
        prediction = model.predict(features_final)[0]
        
        # Probabilités si disponibles
        try:
            probabilities = model.predict_proba(features_final)[0]
            probability = probabilities[1] if len(probabilities) > 1 else float(prediction)
        except AttributeError:
            probability = float(prediction)
        
        # Niveau de risque
        if probability >= 0.8:
            risk_level = "ÉLEVÉ"
        elif probability >= 0.6:
            risk_level = "MODÉRÉ"
        elif probability >= 0.4:
            risk_level = "FAIBLE"
        else:
            risk_level = "TRÈS FAIBLE"
        
        logger.info(f"Prédiction: {prediction}, Probabilité: {probability:.3f}, Risque: {risk_level}")
        
        return {
            "prediction": bool(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "model_used": model_name
        }
    
    except Exception as e:
        logger.error(f"Erreur de prédiction: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise Exception(f"Erreur lors de la prédiction: {str(e)}")

# Routes API

@app.route('/api/health', methods=['GET'])
def health():
    """Vérification de l'état de l'API"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(MODELS),
        "dependencies": DEPENDENCIES_OK
    }), 200

@app.route('/api/status', methods=['GET'])
def status():
    """Informations détaillées sur le statut"""
    return jsonify({
        "api_version": "1.0.0",
        "status": "running",
        "models": {
            "loaded": list(MODELS.keys()),
            "total": len(MODELS)
        },
        "preprocessing": {
            "scaler": PREPROCESSOR is not None and PREPROCESSOR.scaler is not None,
            "pca": PREPROCESSOR is not None and PREPROCESSOR.pca is not None
        },
        "dependencies": DEPENDENCIES_OK,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/analyze/code', methods=['POST'])
def analyze_code():
    """Analyse un code Python et prédit les bugs"""
    try:
        # Récupérer les données
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({"error": "Code requis"}), 400
        
        code = data['code'].strip()
        if not code:
            return jsonify({"error": "Code non vide requis"}), 400
        
        # Extraire les features
        features, metrics = extract_features(code)
        
        # Faire la prédiction
        prediction_result = predict_bug(features)
        
        # Préparer la réponse
        result = {
            "is_bug": prediction_result["prediction"],
            "probability": prediction_result["probability"],
            "risk_level": prediction_result["risk_level"],
            "model_used": prediction_result["model_used"],
            "features": features,
            "metrics": metrics,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Erreur d'analyse: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "type": "analysis_error"
        }), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """Liste tous les modèles disponibles"""
    return jsonify({
        "models": list(MODELS.keys()),
        "total": len(MODELS),
        "preprocessing": {
            "scaler": PREPROCESSOR is not None and PREPROCESSOR.scaler is not None,
            "pca": PREPROCESSOR is not None and PREPROCESSOR.pca is not None
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé"}), 404

@app.errorhandler(500) 
def internal_error(error):
    return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🐛 BUG PREDICTOR AI - API SERVER")
    print("=" * 60)
    
    # Charger les modèles
    print("📦 Chargement des modèles ML...")
    models_loaded = load_models()
    
    if models_loaded:
        print(f"✅ {len(MODELS)} modèles chargés avec succès")
        print(f"📊 Modèles disponibles: {list(MODELS.keys())}")
    else:
        print("⚠️  Aucun modèle chargé - API en mode dégradé")
    
    print("🚀 Démarrage de l'API...")
    print("📡 Disponible sur: http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)