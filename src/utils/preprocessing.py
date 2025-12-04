import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BugPreprocessor:
    """Classe principale pour le prétraitement des données de bugs"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialise le prétraiteur avec configuration
        
        Args:
            config: Configuration du prétraitement
        """
        self.config = config or {
            'test_size': 0.2,
            'random_state': 42,
            'smote_strategy': 'auto',
            'n_pca_components': None,  # None pour auto-selection
            'scale_features': True,
            'apply_smote': True,
            'apply_pca': True
        }
        
        self.scaler = None
        self.pca = None
        self.label_encoder = None
        self.feature_names = None
        
    def apply_smote(self, X: pd.DataFrame, Y: pd.DataFrame, 
                   random_state: int = 42, 
                   sampling_strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applique SMOTE pour équilibrer le dataset
        
        Args:
            X: Features
            Y: Target
            random_state: Seed pour reproductibilité
            sampling_strategy: Stratégie d'oversampling
            
        Returns:
            X_res, Y_res équilibrés
        """
        print("🔄 Application de SMOTE...")
        
        sm = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        X_res, Y_res = sm.fit_resample(X, Y)
        
        Y_res = pd.DataFrame(Y_res, columns=Y.columns)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        
        print(f"   ✓ Avant SMOTE: {X.shape[0]} échantillons")
        print(f"   ✓ Après SMOTE: {X_res.shape[0]} échantillons")
        print(f"   ✓ Distribution: {Y_res.iloc[:, 0].value_counts().to_dict()}")
        
        return X_res, Y_res
    
    def encode_labels(self, Y_df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder]:
        """
        Encode les labels cibles
        
        Args:
            Y_df: DataFrame des labels
            
        Returns:
            y_encoded, label_encoder
        """
        print("🔤 Encodage des labels...")
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(Y_df.values.ravel())
        
        unique_labels = np.unique(y_encoded)
        counts = np.bincount(y_encoded)
        
        print(f"   ✓ Classes: {unique_labels}")
        print(f"   ✓ Distribution: {dict(zip(unique_labels, counts))}")
        
        return y_encoded, self.label_encoder
    
    def split_data(self, X_arr: np.ndarray, y_arr: np.ndarray, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split les données en train/test
        
        Args:
            X_arr: Features array
            y_arr: Labels array
            test_size: Proportion test
            random_state: Seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"✂️  Split des données (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_arr
        )
        
        print(f"   ✓ Train: {X_train.shape[0]} échantillons")
        print(f"   ✓ Test: {X_test.shape[0]} échantillons")
        print(f"   ✓ Distribution train: {np.bincount(y_train)}")
        print(f"   ✓ Distribution test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """
        Normalise les features
        
        Args:
            X_train: Train features
            X_test: Test features
            
        Returns:
            X_train_scaled, X_test_scaled, scaler
        """
        print("📊 Normalisation des features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   ✓ Train scaled: {X_train_scaled.shape}")
        print(f"   ✓ Test scaled: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, self.scaler
    
    def apply_pca(self, X_train: np.ndarray, X_test: np.ndarray, 
                 n_components: Optional[int] = None, variance_threshold: float = 0.95) -> Tuple:
        """
        Applique PCA pour réduction de dimension
        
        Args:
            X_train: Train features
            X_test: Test features
            n_components: Nombre de composants (None pour auto)
            variance_threshold: Seuil de variance expliquée
            
        Returns:
            X_train_pca, X_test_pca, pca
        """
        print("🔍 Application de PCA...")
        
        if n_components is None:
            # PCA avec variance expliquée
            pca = PCA(n_components=variance_threshold)
            X_train_pca = pca.fit_transform(X_train)
            n_components = pca.n_components_
        else:
            # PCA avec nombre fixe de composants
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
        
        X_test_pca = pca.transform(X_test)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        
        print(f"   ✓ Réduction: {X_train.shape[1]} → {n_components} composants")
        print(f"   ✓ Variance expliquée: {explained_variance:.2%}")
        print(f"   ✓ Composants importants: {pca.explained_variance_ratio_[:5].round(3)}...")
        
        self.pca = pca
        return X_train_pca, X_test_pca, pca
    
    def save_preprocessors(self, path: str = 'models/'):
        """
        Sauvegarde les préprocesseurs
        
        Args:
            path: Chemin de sauvegarde
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.scaler:
            joblib.dump(self.scaler, f'{path}scaler.pkl')
        if self.pca:
            joblib.dump(self.pca, f'{path}pca.pkl')
        if self.label_encoder:
            joblib.dump(self.label_encoder, f'{path}label_encoder.pkl')
        
        # Sauvegarder la configuration
        config_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'saved_at': pd.Timestamp.now().isoformat()
        }
        joblib.dump(config_data, f'{path}preprocessor_config.pkl')
        
        print(f"💾 Préprocesseurs sauvegardés dans {path}")
    
    def load_preprocessors(self, path: str = 'models/') -> Tuple:
        """
        Charge les préprocesseurs
        
        Args:
            path: Chemin des préprocesseurs
            
        Returns:
            scaler, pca, label_encoder
        """
        try:
            self.scaler = joblib.load(f'{path}scaler.pkl')
            self.pca = joblib.load(f'{path}pca.pkl')
            self.label_encoder = joblib.load(f'{path}label_encoder.pkl')
            
            # Charger la configuration
            config_data = joblib.load(f'{path}preprocessor_config.pkl')
            self.config = config_data.get('config', {})
            self.feature_names = config_data.get('feature_names', [])
            
            print(f"📂 Préprocesseurs chargés depuis {path}")
            print(f"   Configuration: {self.config}")
            
            return self.scaler, self.pca, self.label_encoder
            
        except Exception as e:
            print(f"⚠️ Erreur chargement préprocesseurs: {e}")
            return None, None, None
    
    def full_preprocessing_pipeline(self, X: pd.DataFrame, Y: pd.DataFrame) -> dict:
        """
        Pipeline complet de prétraitement
        
        Args:
            X: Features
            Y: Target
            
        Returns:
            Dictionnaire avec toutes les données prétraitées
        """
        print("=" * 60)
        print("🚀 DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT")
        print("=" * 60)
        
        # Sauvegarder les noms des features
        self.feature_names = X.columns.tolist()
        print(f"📋 Features: {len(self.feature_names)}")
        
        results = {
            'original_X': X,
            'original_Y': Y,
            'feature_names': self.feature_names
        }
        
        # 1. SMOTE (optionnel)
        if self.config.get('apply_smote', True):
            X_res, Y_res = self.apply_smote(
                X, Y,
                random_state=self.config.get('random_state', 42),
                sampling_strategy=self.config.get('smote_strategy', 'auto')
            )
        else:
            X_res, Y_res = X, Y
            print("⚠️ SMOTE désactivé")
        
        results['X_resampled'] = X_res
        results['Y_resampled'] = Y_res
        
        # 2. Encodage des labels
        y_encoded, label_encoder = self.encode_labels(Y_res)
        results['label_encoder'] = label_encoder
        
        # 3. Split des données
        X_train, X_test, y_train, y_test = self.split_data(
            X_res.values, y_encoded,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42)
        )
        
        results.update({
            'X_train_raw': X_train,
            'X_test_raw': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        
        # 4. Normalisation (optionnel)
        if self.config.get('scale_features', True):
            X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
            results['scaler'] = scaler
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
            print("⚠️ Normalisation désactivée")
        
        results.update({
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled
        })
        
        # 5. PCA (optionnel)
        if self.config.get('apply_pca', True):
            n_components = self.config.get('n_pca_components')
            X_train_pca, X_test_pca, pca = self.apply_pca(
                X_train_scaled, X_test_scaled,
                n_components=n_components
            )
            results['pca'] = pca
            X_train_final, X_test_final = X_train_pca, X_test_pca
        else:
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
            print("⚠️ PCA désactivé")
        
        results.update({
            'X_train_final': X_train_final,
            'X_test_final': X_test_final
        })
        
        # 6. Sauvegarde
        if self.config.get('save_preprocessors', True):
            self.save_preprocessors()
        
        print("\n" + "=" * 60)
        print("✅ PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        
        summary = {
            'samples_total': len(X),
            'samples_after_smote': len(X_res),
            'train_samples': len(X_train_final),
            'test_samples': len(X_test_final),
            'original_features': X.shape[1],
            'final_features': X_train_final.shape[1],
            'feature_reduction': f"{X.shape[1] - X_train_final.shape[1]} features",
            'class_distribution': {
                'train': np.bincount(y_train).tolist(),
                'test': np.bincount(y_test).tolist()
            }
        }
        
        print("\n📊 RÉSUMÉ:")
        for key, value in summary.items():
            print(f"   • {key}: {value}")
        
        return results
    
    def transform_new_data(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Transforme de nouvelles données avec les préprocesseurs entraînés
        
        Args:
            X_new: Nouvelles données
            
        Returns:
            Données transformées
        """
        if self.scaler is None or self.pca is None:
            raise ValueError("Préprocesseurs non chargés. Appelez load_preprocessors() d'abord.")
        
        print(f"🔄 Transformation de {len(X_new)} nouveaux échantillons...")
        
        # Vérifier les features
        if self.feature_names and set(X_new.columns) != set(self.feature_names):
            print(f"⚠️ Features différentes. Attendu: {self.feature_names[:5]}...")
            # Réorganiser les colonnes si nécessaire
            missing = set(self.feature_names) - set(X_new.columns)
            extra = set(X_new.columns) - set(self.feature_names)
            if missing:
                print(f"   ❌ Features manquantes: {list(missing)[:5]}...")
                raise ValueError(f"Features manquantes: {list(missing)}")
        
        # Transformation
        X_scaled = self.scaler.transform(X_new.values)
        X_pca = self.pca.transform(X_scaled)
        
        print(f"   ✓ Shape finale: {X_pca.shape}")
        return X_pca


# Fonctions utilitaires (compatibilité avec ancien code)
def apply_smote(X, Y, random_state=12, sampling_strategy=1.0):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor()
    X_res, Y_res = preprocessor.apply_smote(X, Y, random_state, sampling_strategy)
    return X_res, Y_res

def encode_labels(Y_df):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor()
    y_encoded, label_encoder = preprocessor.encode_labels(Y_df)
    return y_encoded, label_encoder

def split_data(X_arr, y_arr, test_size=0.25, random_state=0):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor({'test_size': test_size, 'random_state': random_state})
    return preprocessor.split_data(X_arr, y_arr, test_size, random_state)

def scale_features(X_train, X_test):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor()
    return preprocessor.scale_features(X_train, X_test)

def apply_pca(X_train, X_test, n_components=6):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor({'n_pca_components': n_components})
    return preprocessor.apply_pca(X_train, X_test, n_components)

def save_preprocessors(scaler, pca, path='models/'):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor()
    preprocessor.scaler = scaler
    preprocessor.pca = pca
    preprocessor.save_preprocessors(path)

def load_preprocessors(path='models/'):
    """Fonction wrapper pour compatibilité"""
    preprocessor = BugPreprocessor()
    scaler, pca, _ = preprocessor.load_preprocessors(path)
    return scaler, pca


if __name__ == '__main__':
    # Test du pipeline
    print("🧪 Test du prétraitement...")
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Créer des labels déséquilibrés
    y_test = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    Y_test = pd.DataFrame(y_test, columns=['target'])
    
    print(f"Données de test: {X_test.shape}")
    print(f"Distribution originale: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Utiliser le pipeline complet
    preprocessor = BugPreprocessor({
        'test_size': 0.2,
        'random_state': 42,
        'apply_smote': True,
        'apply_pca': True,
        'n_pca_components': 10,
        'save_preprocessors': False
    })
    
    results = preprocessor.full_preprocessing_pipeline(X_test, Y_test)
    
    print("\n✅ Test réussi!")