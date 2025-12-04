"""
Decision Tree Model
"""

from sklearn import tree
import time
from .model import BaseModel


class DecisionTreeModel(BaseModel):
    """Modèle Decision Tree"""
    
    def __init__(self, max_depth=None, **kwargs):
        """
        Initialise le modèle Decision Tree
        
        Args:
            max_depth: Profondeur maximale
            **kwargs: Paramètres supplémentaires
        """
        super().__init__("DecisionTree", max_depth=max_depth, **kwargs)
        self.build_model()
    
    def build_model(self):
        """Construit le modèle Decision Tree"""
        self.model = tree.DecisionTreeClassifier(
            criterion=self.params.get('criterion', 'gini'),
            splitter=self.params.get('splitter', 'best'),
            max_depth=self.params.get('max_depth'),
            min_samples_split=self.params.get('min_samples_split', 2),
            min_samples_leaf=self.params.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=self.params.get('min_weight_fraction_leaf', 0.0),
            max_features=self.params.get('max_features', None),
            random_state=self.params.get('random_state', None),
            max_leaf_nodes=self.params.get('max_leaf_nodes', None),
            min_impurity_decrease=self.params.get('min_impurity_decrease', 0.0),
            class_weight=self.params.get('class_weight', None),
            ccp_alpha=self.params.get('ccp_alpha', 0.0)
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Entraîne le modèle Decision Tree
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Arguments supplémentaires
        """
        start_time = time.time()
        
        print("🔧 Entraînement du modèle Decision Tree...")
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.trained = True
        
        print(f"✅ Decision Tree entraîné en {self.training_time:.2f}s")
        return self
    
    def get_feature_importance(self):
        """
        Retourne l'importance des features
        
        Returns:
            Dictionnaire feature -> importance
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné")
        
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names:
                return dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                return {f'feature_{i}': imp for i, imp in enumerate(self.model.feature_importances_)}
        return {}