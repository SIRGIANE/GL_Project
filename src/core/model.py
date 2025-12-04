"""
Classe de base pour tous les modèles de prédiction de bugs
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib
import json
from datetime import datetime
from pathlib import Path


class BaseModel(ABC):
    """Classe abstraite de base pour tous les modèles"""
    
    def __init__(self, model_name: str, **params):
        """
        Initialise le modèle
        
        Args:
            model_name: Nom du modèle
            **params: Paramètres du modèle
        """
        self.model_name = model_name
        self.params = params
        self.model = None
        self.trained = False
        self.training_time = None
        self.metrics = {}
        self.feature_names = None
        
    @abstractmethod
    def build_model(self):
        """Construit l'architecture du modèle"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Entraîne le modèle
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Arguments supplémentaires
        """
        pass
    
    def predict(self, X):
        """
        Prédit les classes
        
        Args:
            X: Features à prédire
            
        Returns:
            Prédictions
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Prédit les probabilités
        
        Args:
            X: Features à prédire
            
        Returns:
            Probabilités
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback pour les modèles sans predict_proba
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            for i, pred in enumerate(predictions):
                proba[i, int(pred)] = 1.0
            return proba
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            
        Returns:
            Métriques d'évaluation
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        if y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba[:, 0]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return self.metrics
    
    def save(self, filepath: str):
        """
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin du fichier
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné. Impossible de sauvegarder.")
        
        # Sauvegarder le modèle
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'params': self.params,
            'trained': self.trained,
            'training_time': self.training_time,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """
        Charge le modèle
        
        Args:
            filepath: Chemin du fichier
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.params = model_data['params']
        self.trained = model_data['trained']
        self.training_time = model_data['training_time']
        self.metrics = model_data.get('metrics', {})
        self.feature_names = model_data.get('feature_names', [])
        
        print(f"✅ Modèle chargé: {self.model_name}")
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé du modèle
        
        Returns:
            Dictionnaire avec les informations du modèle
        """
        return {
            'model_name': self.model_name,
            'params': self.params,
            'trained': self.trained,
            'training_time': self.training_time,
            'metrics': self.metrics,
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }
    
    def set_feature_names(self, feature_names):
        """Définit les noms des features"""
        self.feature_names = feature_names
    
    def __str__(self):
        return f"{self.model_name} (trained: {self.trained})"