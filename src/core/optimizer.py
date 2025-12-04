from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import joblib

class ModelOptimizer:
    """Optimise les hyperparamètres des modèles"""
    
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid
        self.best_model = None
        self.best_params = None
        self.best_score = None
        
    def optimize_grid_search(self, X, y, cv=5, scoring='f1', n_jobs=-1):
        """Optimisation par Grid Search"""
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"Meilleurs paramètres: {self.best_params}")
        print(f"Meilleur score ({scoring}): {self.best_score:.4f}")
        
        return self.best_model
    
    def optimize_random_search(self, X, y, cv=5, scoring='f1', 
                              n_iter=50, n_jobs=-1):
        """Optimisation par Random Search"""
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        print(f"Meilleurs paramètres: {self.best_params}")
        print(f"Meilleur score ({scoring}): {self.best_score:.4f}")
        
        return self.best_model
    
    def save_best_model(self, path):
        """Sauvegarde le meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle optimisé à sauvegarder")
        
        joblib.dump(self.best_model, path)
        print(f"Meilleur modèle sauvegardé à {path}")
    
    @staticmethod
    def get_default_param_grids():
        """Retourne les grilles de paramètres par défaut pour chaque modèle"""
        param_grids = {
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 500, 1000]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1],
                'degree': [2, 3, 4]
            }
        }
        
        return param_grids