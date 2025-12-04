"""
Factory pour créer des instances de modèles
"""

from .knn import KNNModel
from .naive_bayes import NaiveBayesModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel
from .svm import SVMModel
from .logistic_regression import LogisticRegressionModel

from .lstm import LSTMModel

from .model import BaseModel


class ModelFactory:
    """Factory pour créer des instances de modèles"""
    
    @staticmethod
    def create_model(model_type: str, **params) -> BaseModel:
        """
        Crée une instance de modèle
        
        Args:
            model_type: Type de modèle
            **params: Paramètres du modèle
            
        Returns:
            Instance du modèle
        """
        model_registry = {
            'knn': KNNModel,
            'naive_bayes': NaiveBayesModel,
            'decision_tree': DecisionTreeModel,
            'random_forest': RandomForestModel,
            'svm': SVMModel,
            'logistic_regression': LogisticRegressionModel,
            
            'lstm': LSTMModel
        }
        
        if model_type not in model_registry:
            raise ValueError(f"Type de modèle non supporté: {model_type}. "
                           f"Types disponibles: {list(model_registry.keys())}")
        
        return model_registry[model_type](**params)
    
    @staticmethod
    def get_available_models():
        """Retourne la liste des modèles disponibles"""
        return {
            'knn': 'K-Nearest Neighbors',
            'naive_bayes': 'Gaussian Naive Bayes',
            'decision_tree': 'Decision Tree',
            'random_forest': 'Random Forest',
            'svm': 'Support Vector Machine',
            'logistic_regression': 'Logistic Regression',
            'neural_network': 'Neural Network',
            'lstm': 'Long Short-Term Memory',
            'cnn': 'Convolutional Neural Network'
        }