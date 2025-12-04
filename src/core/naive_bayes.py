"""
Naive Bayes Model
"""

from sklearn.naive_bayes import GaussianNB
import time
from .model import BaseModel


class NaiveBayesModel(BaseModel):
    """Modèle Gaussian Naive Bayes"""
    
    def __init__(self, **kwargs):
        """
        Initialise le modèle Naive Bayes
        
        Args:
            **kwargs: Paramètres du modèle
        """
        super().__init__("NaiveBayes", **kwargs)
        self.build_model()
    
    def build_model(self):
        """Construit le modèle Naive Bayes"""
        self.model = GaussianNB(
            var_smoothing=self.params.get('var_smoothing', 1e-9)
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Entraîne le modèle Naive Bayes
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Arguments supplémentaires
        """
        start_time = time.time()
        
        print("🔧 Entraînement du modèle Naive Bayes...")
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.trained = True
        
        print(f"✅ Naive Bayes entraîné en {self.training_time:.2f}s")
        return self