
"""
Pipeline d'entraînement des modèles de prédiction de bugs
"""

import sys
import os
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add the project root to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.dataset import load_and_preprocess_data
from src.utils.preprocessing import (
    apply_smote, encode_labels, split_data,
    scale_features, apply_pca, save_preprocessors
)

# Import des modèles depuis la nouvelle architecture
from src.core.model_factory import ModelFactory
from src.core.model import BaseModel

# Define paths
MODELS_DIR = 'bug-predictor/models/'
RESULTS_DIR = 'bug-predictor/results/'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_models_selected(
    data_path: str = 'GL_Project/data/',
    models_to_train: List[str] = None,
    use_smote: bool = True,
    use_pca: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, BaseModel], Dict[str, Any]]:
    """
    Entraîne les modèles sélectionnés
    
    Args:
        data_path: Chemin vers les données
        models_to_train: Liste des modèles à entraîner
        use_smote: Appliquer SMOTE pour équilibrer les classes
        use_pca: Appliquer PCA pour réduire la dimension
        test_size: Proportion des données de test
        random_state: Seed pour la reproductibilité
        
    Returns:
        Tuple contenant:
            - Dictionnaire des modèles entraînés
            - Dictionnaire des métriques et préprocesseurs
    """
    print("🚀 Démarrage du pipeline d'entraînement...")
    print(f"📋 Modèles à entraîner: {models_to_train}")
    
    # Liste par défaut des modèles si non spécifiée
    if models_to_train is None:
        models_to_train = [
            'LogisticRegressionModel',
            'LSTMModel',
            'RandomForestModel',
            'SVMModel',
            'KNNModel',
            'NaiveBayesModel',
            'DecisionTreeModel'
        ]
    
    # 1. Chargement et prétraitement des données
    print("\n📥 1. Chargement des données...")
    X_df, Y_df, _ = load_and_preprocess_data(data_path=data_path)
    print(f"   ✓ Données chargées: {X_df.shape[0]} échantillons, {X_df.shape[1]} features")
    
    # 2. Application de SMOTE (optionnel)
    if use_smote:
        print("\n⚖️  2. Application de SMOTE pour équilibrer les classes...")
        X_res, Y_res = apply_smote(X_df, Y_df)
        print(f"   ✓ Données après SMOTE: {X_res.shape[0]} échantillons")
    else:
        X_res, Y_res = X_df, Y_df
        print("\n⚖️  2. SMOTE désactivé")
    
    # 3. Encodage des labels
    print("\n🔤 3. Encodage des labels...")
    y_encoded, label_encoder = encode_labels(Y_res)
    class_distribution = np.bincount(y_encoded)
    print(f"   ✓ Classe 0: {class_distribution[0]} échantillons")
    print(f"   ✓ Classe 1: {class_distribution[1]} échantillons")
    
    # 4. Split des données
    print(f"\n✂️  4. Split des données (test_size={test_size})...")
    X_train, X_test, y_train, y_test = split_data(
        X_res.values, y_encoded, test_size=test_size, random_state=random_state
    )
    print(f"   ✓ Train: {X_train.shape[0]} échantillons")
    print(f"   ✓ Test: {X_test.shape[0]} échantillons")
    
    # 5. Normalisation des features
    print("\n📊 5. Normalisation des features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("   ✓ Features normalisées")
    
    # 6. Application de PCA (optionnel)
    if use_pca:
        print("\n🔍 6. Application de PCA...")
        X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
        print(f"   ✓ Réduction de dimension: {X_train.shape[1]} → {X_train_pca.shape[1]} features")
        X_train_final, X_test_final = X_train_pca, X_test_pca
    else:
        X_train_final, X_test_final = X_train_scaled, X_test_scaled
        pca = None
        print("\n🔍 6. PCA désactivé")
    
    # Sauvegarde des préprocesseurs
    print("\n💾 7. Sauvegarde des préprocesseurs...")
    save_preprocessors(scaler, pca, path=MODELS_DIR)
    print("   ✓ Préprocesseurs sauvegardés")
    
    # Initialisation des résultats
    trained_models = {}
    all_metrics = {}
    training_summary = []
    
    print(f"\n🤖 8. Entraînement des modèles ({len(models_to_train)} modèles)...")
    
    # Paramètres par défaut pour chaque modèle
    default_params = {
        'LogisticRegressionModel': {
            'C': 1.0,
            'max_iter': 1000,
            'calibrate': True,
            'random_state': random_state
        },
        'LSTMModel': {
            'units': 100,
            'dropout_rate': 0.2,
            'input_dim': X_train_final.shape[1]
        },
        'RandomForestModel': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': random_state
        },
        'SVMModel': {
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True,
            'random_state': random_state
        },
        'KNNModel': {
            'n_neighbors': 5,
            'weights': 'uniform'
        },
        'NaiveBayesModel': {
            'var_smoothing': 1e-9
        },
        'DecisionTreeModel': {
            'max_depth': None,
            'random_state': random_state
        }
    }
    
    # Entraînement de chaque modèle
    for model_name in models_to_train:
        try:
            print(f"\n   🔧 {model_name}...")
            
            # Création du modèle avec les paramètres par défaut
            params = default_params.get(model_name, {})
            model = ModelFactory.create_model(model_name.lower().replace('model', ''), **params)
            
            # Définir les noms des features
            if hasattr(model, 'set_feature_names'):
                feature_names = [f'feature_{i}' for i in range(X_train_final.shape[1])]
                model.set_feature_names(feature_names)
            
            # Entraînement spécifique pour LSTM
            if model_name == 'LSTMModel':
                model.train(X_train_final, y_train, epochs=50, batch_size=32)
            else:
                model.train(X_train_final, y_train)
            
            # Évaluation
            metrics = model.evaluate(X_test_final, y_test)
            
            # Stockage des résultats
            trained_models[model_name] = model
            all_metrics[model_name] = metrics
            
            # Sauvegarde du modèle
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{MODELS_DIR}{model_name}_{timestamp}.joblib"
            
            # Gestion spéciale pour les modèles Keras
            if model_name in ['LSTMModel']:
                keras_filename = f"{MODELS_DIR}{model_name}_{timestamp}.h5"
                model.model.save(keras_filename)
                print(f"     💾 Modèle Keras sauvegardé: {keras_filename}")
            
            model.save(model_filename)
            print(f"     💾 Modèle sauvegardé: {model_filename}")
            
            # Ajout au résumé
            training_summary.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
                'Training Time (s)': f"{model.training_time:.2f}"
            })
            
            print(f"     ✅ Performance: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
            
        except Exception as e:
            print(f"     ❌ Erreur avec {model_name}: {str(e)}")
            continue
    
    # 9. Analyse comparative
    print("\n📊 9. Analyse comparative des modèles...")
    
    if training_summary:
        # Trier par F1-Score
        training_summary.sort(key=lambda x: float(x['F1-Score']), reverse=True)
        
        print("\n🏆 CLASSEMENT DES MODÈLES (par F1-Score):")
        print("-" * 80)
        for i, model_info in enumerate(training_summary):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
            print(f"{rank} {model_info['Model']:25s}")
            print(f"   F1-Score: {model_info['F1-Score']} | Accuracy: {model_info['Accuracy']} | ROC-AUC: {model_info['ROC-AUC']}")
            print(f"   Training Time: {model_info['Training Time (s)']}s")
            print()
    
    # 10. Sauvegarde des résultats
    print("\n💾 10. Sauvegarde des résultats...")
    
    # Sauvegarde des métriques
    import pandas as pd
    import json
    
    metrics_df = pd.DataFrame(training_summary)
    metrics_csv = f"{RESULTS_DIR}model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"   ✓ Métriques sauvegardées: {metrics_csv}")
    
    # Sauvegarde des métriques détaillées en JSON
    metrics_json = f"{RESULTS_DIR}detailed_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"   ✓ Métriques détaillées sauvegardées: {metrics_json}")
    
    # Sauvegarde du résumé d'entraînement
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': models_to_train,
        'dataset_info': {
            'original_samples': X_df.shape[0],
            'final_train_samples': X_train_final.shape[0],
            'test_samples': X_test_final.shape[0],
            'n_features_original': X_df.shape[1],
            'n_features_final': X_train_final.shape[1],
            'use_smote': use_smote,
            'use_pca': use_pca,
            'test_size': test_size,
            'random_state': random_state
        },
        'best_model': training_summary[0]['Model'] if training_summary else None,
        'best_f1_score': training_summary[0]['F1-Score'] if training_summary else None
    }
    
    summary_json = f"{RESULTS_DIR}training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Résumé d'entraînement sauvegardé: {summary_json}")
    
    # 11. Retour des résultats
    results = {
        'trained_models': trained_models,
        'metrics': all_metrics,
        'training_summary': training_summary,
        'preprocessors': {
            'scaler': scaler,
            'pca': pca,
            'label_encoder': label_encoder
        },
        'data': {
            'X_train': X_train_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_test': y_test
        }
    }
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    
    print(f"\n📊 RÉSUMÉ FINAL:")
    print(f"   • Modèles entraînés: {len(trained_models)}/{len(models_to_train)}")
    print(f"   • Meilleur modèle: {summary['best_model']}")
    print(f"   • Meilleur F1-Score: {summary['best_f1_score']}")
    print(f"   • Données d'entraînement: {X_train_final.shape}")
    print(f"   • Données de test: {X_test_final.shape}")
    
    return trained_models, results

def train_single_model(
    model_type: str,
    data_path: str = 'bug-predictor/data/',
    model_params: Dict[str, Any] = None,
    **training_kwargs
) -> Tuple[BaseModel, Dict[str, Any]]:
    """
    Entraîne un seul modèle spécifique
    
    Args:
        model_type: Type de modèle à entraîner
        data_path: Chemin vers les données
        model_params: Paramètres spécifiques du modèle
        **training_kwargs: Arguments d'entraînement supplémentaires
        
    Returns:
        Tuple (modèle entraîné, métriques)
    """
    print(f"🚀 Entraînement du modèle {model_type}...")
    
    # Chargement des données (réutilise la logique de train_models_selected)
    X_df, Y_df, _ = load_and_preprocess_data(data_path=data_path)
    X_res, Y_res = apply_smote(X_df, Y_df)
    y_encoded, label_encoder = encode_labels(Y_res)
    X_train, X_test, y_train, y_test = split_data(
        X_res.values, y_encoded, test_size=0.2, random_state=42
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
    
    # Création du modèle
    if model_params is None:
        model_params = {}
    
    model = ModelFactory.create_model(model_type, **model_params)
    
    # Définir les noms des features
    if hasattr(model, 'set_feature_names'):
        feature_names = [f'feature_{i}' for i in range(X_train_pca.shape[1])]
        model.set_feature_names(feature_names)
    
    # Entraînement
    if model_type == 'lstm':
        model.train(X_train_pca, y_train, **training_kwargs)
    else:
        model.train(X_train_pca, y_train, **training_kwargs)
    
    # Évaluation
    metrics = model.evaluate(X_test_pca, y_test)
    
    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{MODELS_DIR}{model_type}_{timestamp}.joblib"
    model.save(model_filename)
    
    print(f"✅ {model_type} entraîné avec succès!")
    print(f"📊 Performance: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
    print(f"💾 Modèle sauvegardé: {model_filename}")
    
    return model, metrics

def load_trained_model(model_path: str) -> BaseModel:
    """
    Charge un modèle entraîné précédemment
    
    Args:
        model_path: Chemin vers le fichier du modèle
        
    Returns:
        Modèle chargé
    """
    print(f"📂 Chargement du modèle: {model_path}")
    
    model = BaseModel("", {})
    model.load(model_path)
    
    print(f"✅ Modèle chargé: {model.model_name}")
    print(f"   Entraîné le: {model.training_time}")
    print(f"   Métriques: {model.metrics}")
    
    return model

def compare_models(
    model_paths: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare plusieurs modèles pré-entraînés
    
    Args:
        model_paths: Liste des chemins vers les modèles
        X_test: Données de test
        y_test: Labels de test
        
    Returns:
        Dictionnaire des métriques pour chaque modèle
    """
    print("📊 Comparaison de modèles...")
    
    comparison_results = {}
    
    for model_path in model_paths:
        try:
            # Charger le modèle
            model = load_trained_model(model_path)
            
            # Évaluer
            metrics = model.evaluate(X_test, y_test)
            
            comparison_results[model.model_name] = {
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'roc_auc': metrics.get('roc_auc', 0),
                'training_time': model.training_time
            }
            
            print(f"   ✓ {model.model_name}: F1={metrics['f1_score']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Erreur avec {model_path}: {e}")
            continue
    
    # Trier par F1-Score
    sorted_results = dict(sorted(
        comparison_results.items(),
        key=lambda x: x[1]['f1_score'],
        reverse=True
    ))
    
    print("\n🏆 CLASSEMENT:")
    for i, (model_name, metrics) in enumerate(sorted_results.items()):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank} {model_name:20s} | F1: {metrics['f1_score']:.3f} | Acc: {metrics['accuracy']:.3f}")
    
    return sorted_results

if __name__ == '__main__':
    """
    Exemple d'utilisation
    """
    
    # Option 1: Entraîner tous les modèles par défaut
    trained_models, results = train_models_selected()
    
    # Option 2: Entraîner seulement certains modèles
    # selected_models = ['RandomForestModel', 'LogisticRegressionModel', 'SVMModel']
    # trained_models, results = train_models_selected(models_to_train=selected_models)
    
    # Option 3: Entraîner un seul modèle
    # model, metrics = train_single_model(
    #     model_type='random_forest',
    #     model_params={'n_estimators': 200, 'max_depth': 15},
    #     data_path='bug-predictor/data/'
    # )
    
    # Option 4: Comparer des modèles existants
    # model_files = [
    #     'bug-predictor/models/RandomForestModel_20250101_120000.joblib',
    #     'bug-predictor/models/LogisticRegressionModel_20250101_120000.joblib'
    # ]
    # comparison = compare_models(model_files, results['data']['X_test'], results['data']['y_test'])