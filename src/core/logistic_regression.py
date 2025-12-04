"""
Logistic Regression Model
"""

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import time
from .model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Modèle Logistic Regression avec calibration"""
    
    def __init__(self, C=1.0, max_iter=1000, calibrate=True, **kwargs):
        """
        Initialise le modèle Logistic Regression
        
        Args:
            C: Paramètre de régularisation
            max_iter: Nombre maximal d'itérations
            calibrate: Si True, calibre les probabilités
            **kwargs: Paramètres supplémentaires
        """
        super().__init__("LogisticRegression", C=C, max_iter=max_iter, calibrate=calibrate, **kwargs)
        self.calibrate = calibrate
        self.build_model()
    
    def build_model(self):
        """Construit le modèle Logistic Regression"""
        base_model = LogisticRegression(
            penalty=self.params.get('penalty', 'l2'),
            dual=self.params.get('dual', False),
            tol=self.params.get('tol', 1e-4),
            C=self.params.get('C', 1.0),
            fit_intercept=self.params.get('fit_intercept', True),
            intercept_scaling=self.params.get('intercept_scaling', 1),
            class_weight=self.params.get('class_weight', None),
            random_state=self.params.get('random_state', None),
            solver=self.params.get('solver', 'lbfgs'),
            max_iter=self.params.get('max_iter', 100),
            multi_class=self.params.get('multi_class', 'auto'),
            verbose=self.params.get('verbose', 0),
            warm_start=self.params.get('warm_start', False),
            n_jobs=self.params.get('n_jobs', None),
            l1_ratio=self.params.get('l1_ratio', None)
        )
        
        if self.calibrate:
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                cv=self.params.get('cv', 3),
                method=self.params.get('method', 'sigmoid')
            )
        else:
            self.model = base_model
    
    def train(self, X_train, y_train, **kwargs):
        """
        Entraîne le modèle Logistic Regression
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Arguments supplémentaires
        """
        start_time = time.time()
        
        print(f"🔧 Entraînement du modèle Logistic Regression (C={self.params['C']}, calibrate={self.calibrate})...")
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.trained = True
        
        print(f"✅ Logistic Regression entraîné en {self.training_time:.2f}s")
        return self
    
    def get_coefficients(self):
        """
        Retourne les coefficients du modèle
        
        Returns:
            Dictionnaire feature -> coefficient
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné")
        
        if hasattr(self.model, 'coef_'):
            if self.calibrate:
                # Pour CalibratedClassifierCV, accéder au modèle de base
                base_model = self.model.estimator
                if hasattr(base_model, 'coef_'):
                    coefs = base_model.coef_[0]
                else:
                    return {}
            else:
                coefs = self.model.coef_[0]
            
            if self.feature_names:
                return dict(zip(self.feature_names, coefs))
            else:
                return {f'feature_{i}': coef for i, coef in enumerate(coefs)}
        return {}