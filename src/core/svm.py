"""
Support Vector Machine Model
"""

from sklearn.svm import SVC
import time
from .model import BaseModel


class SVMModel(BaseModel):
    """Modèle Support Vector Machine"""
    
    def __init__(self, kernel='rbf', C=1.0, **kwargs):
        """
        Initialise le modèle SVM
        
        Args:
            kernel: Type de kernel
            C: Paramètre de régularisation
            **kwargs: Paramètres supplémentaires
        """
        super().__init__("SVM", kernel=kernel, C=C, **kwargs)
        self.build_model()
    
    def build_model(self):
        """Construit le modèle SVM"""
        self.model = SVC(
            C=self.params.get('C', 1.0),
            kernel=self.params.get('kernel', 'rbf'),
            degree=self.params.get('degree', 3),
            gamma=self.params.get('gamma', 'scale'),
            coef0=self.params.get('coef0', 0.0),
            shrinking=self.params.get('shrinking', True),
            probability=self.params.get('probability', True),
            tol=self.params.get('tol', 1e-3),
            cache_size=self.params.get('cache_size', 200),
            class_weight=self.params.get('class_weight', None),
            verbose=self.params.get('verbose', False),
            max_iter=self.params.get('max_iter', -1),
            decision_function_shape=self.params.get('decision_function_shape', 'ovr'),
            break_ties=self.params.get('break_ties', False),
            random_state=self.params.get('random_state', None)
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Entraîne le modèle SVM
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Arguments supplémentaires
        """
        start_time = time.time()
        
        print(f"🔧 Entraînement du modèle SVM (kernel={self.params['kernel']}, C={self.params['C']})...")
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.trained = True
        
        print(f"✅ SVM entraîné en {self.training_time:.2f}s")
        return self