from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def evaluate_model_performance(model, X_test, y_test, model_name="Model") -> Dict[str, float]:
    """
    Évalue un modèle et retourne les métriques principales
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
        model_name: Nom du modèle
        
    Returns:
        Dictionnaire des métriques
    """
    print(f"📊 {model_name}")
    print("-" * 40)
    
    # Prédictions selon le type de modèle
    if 'keras' in str(type(model)).lower():
        # Modèles Keras
        if 'lstm' in str(type(model)).lower() or model_name == "LSTM":
            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        elif 'cnn' in str(type(model)).lower() or model_name == "CNN":
            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)
        else:
            X_test_reshaped = X_test
            
        y_pred_proba = model.predict(X_test_reshaped, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
    else:
        # Modèles scikit-learn
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred.astype(float)
    
    # Conversion en int
    y_test_int = y_test.astype(int)
    
    # Calcul des métriques
    metrics = {
        'accuracy': accuracy_score(y_test_int, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test_int, y_pred),
        'precision': precision_score(y_test_int, y_pred, zero_division=0),
        'recall': recall_score(y_test_int, y_pred, zero_division=0),
        'f1_score': f1_score(y_test_int, y_pred, zero_division=0)
    }
    
    # ROC-AUC (si probabilités disponibles)
    try:
        if 'y_pred_proba' in locals():
            metrics['roc_auc'] = roc_auc_score(y_test_int, y_pred_proba)
        else:
            metrics['roc_auc'] = roc_auc_score(y_test_int, y_pred)
    except:
        metrics['roc_auc'] = 0.0
    
    # Matrice de confusion
    cm = confusion_matrix(y_test_int, y_pred)
    
    # Affichage
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"Matrice confusion:\n{cm}")
    
    print("=" * 40)
    
    return metrics

def evaluate_multiple_models(models_dict: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
    """
    Évalue plusieurs modèles et retourne un DataFrame comparatif
    
    Args:
        models_dict: Dictionnaire {nom: modèle}
        X_test: Features de test
        y_test: Labels de test
        
    Returns:
        DataFrame avec les métriques
    """
    results = []
    
    print("🔬 ÉVALUATION COMPARATIVE DES MODÈLES")
    print("=" * 60)
    
    for name, model in models_dict.items():
        print(f"\n📊 {name}")
        print("-" * 30)
        
        metrics = evaluate_model_performance(model, X_test, y_test, name)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    # Création du DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # Affichage du classement
    print("\n🏆 CLASSEMENT DES MODÈLES")
    print("=" * 60)
    for i, row in df.iterrows():
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank} {row['Model']:25s} | F1: {row['F1-Score']:.4f} | Acc: {row['Accuracy']:.4f}")
    
    return df

def get_best_model(models_dict: Dict[str, Any], X_test, y_test, 
                   metric: str = 'f1_score') -> Tuple[str, Any, Dict]:
    """
    Trouve le meilleur modèle selon une métrique
    
    Args:
        models_dict: Dictionnaire des modèles
        X_test: Features de test
        y_test: Labels de test
        metric: Métrique d'évaluation
        
    Returns:
        (nom_du_modèle, modèle, métriques)
    """
    best_model_name = None
    best_model = None
    best_score = -1
    all_metrics = {}
    
    for name, model in models_dict.items():
        metrics = evaluate_model_performance(model, X_test, y_test, name)
        all_metrics[name] = metrics
        
        if metrics[metric] > best_score:
            best_score = metrics[metric]
            best_model_name = name
            best_model = model
    
    print(f"\n🎯 MEILLEUR MODÈLE ({metric}): {best_model_name}")
    print(f"   Score: {best_score:.4f}")
    
    return best_model_name, best_model, all_metrics