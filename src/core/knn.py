"""
K-Nearest Neighbors Model
"""

from sklearn.neighbors import KNeighborsClassifier
import time
from .model import BaseModel


class KNNModel(BaseModel):
    """Modèle K-Nearest Neighbors"""
    
    def __init__(self, n_neighbors=5, **kwargs):
        """
        Initialise le modèle KNN
        
        Args:
            n_neighbors: Nombre de voisins
            **kwargs: Paramètres supplémentaires
        """
        super().__init__("KNN", n_neighbors=n_neighbors, **kwargs)
        self.build_model()
    
    def build_model(self):
        """Construit le modèle KNN"""
        self.model = KNeighborsClassifier(
            n_neighbors=self.params.get('n_neighbors', 5),
            weights=self.params.get('weights', 'uniform'),
            algorithm=self.params.get('algorithm', 'auto'),
            leaf_size=self.params.get('leaf_size', 30),
            p=self.params.get('p', 2),
            metric=self.params.get('metric', 'minkowski')
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Entraîne le modèle KNN
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Arguments supplémentaires
        """
        start_time = time.time()
        
        print(f"🔧 Entraînement du modèle KNN (n_neighbors={self.params['n_neighbors']})...")
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.trained = True
        
        print(f"✅ KNN entraîné en {self.training_time:.2f}s")
        return self