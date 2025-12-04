"""
Module de chargement et fusion des datasets NASA de prédiction de bugs.
Supporte CSV, ARFF, et autres formats tabulaires.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger("bug_predictor.dataset")


class BugDataset:
    """
    Classe pour charger, fusionner et prétraiter les datasets de bugs.
    
    Supporte:
        - Fichiers CSV
        - Fichiers ARFF (scipy.io.arff)
        - Fusion de multiples datasets
        - Application de règles de complexité (Halstead)
    
    Attributes:
        data_path (Path): Chemin vers le dossier contenant les fichiers
        datasets (Dict[str, pd.DataFrame]): Datasets chargés {nom: dataframe}
    """
    
    # Halstead features à garder
    HALSTEAD_FEATURES = ['n', 'v', 'd', 'e', 't']
    # Colonne cible par défaut
    TARGET_COLUMNS = ['defects', 'bugs', 'bug_', 'defect_']
    # Features à supprimer (non significatives)
    FEATURES_TO_DROP = ['ev(g)', 'iv(g)', 'b', 'branchCount', 'id', 'index']
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialise le dataset loader.
        
        Args:
            data_path: Chemin vers dossier contenant les fichiers
            random_state: Graine pour reproducibilité
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.datasets: Dict[str, pd.DataFrame] = {}
        
        if not self.data_path.exists():
            logger.error(f"Dossier data introuvable: {self.data_path}")
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        logger.info(f"BugDataset initialisé pour: {self.data_path}")
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Charge tous les fichiers CSV et ARFF du dossier.
        
        Returns:
            Dict[str, pd.DataFrame]: {nom_fichier: dataframe}
        """
        logger.info("Chargement de tous les datasets...")
        self.datasets = {}
        
        # Charger CSV
        csv_files = list(self.data_path.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                self.datasets[csv_file.stem] = df
                logger.info(f"  ✓ CSV: {csv_file.name} ({df.shape[0]}x{df.shape[1]})")
            except Exception as e:
                logger.warning(f"  ✗ Erreur CSV {csv_file.name}: {e}")
        
        # Charger ARFF si scipy disponible
        try:
            from scipy.io import arff
            arff_files = list(self.data_path.glob("*.arff"))
            for arff_file in arff_files:
                try:
                    data, meta = arff.loadarff(arff_file)
                    df = pd.DataFrame(data)
                    self.datasets[arff_file.stem] = df
                    logger.info(f"  ✓ ARFF: {arff_file.name} ({df.shape[0]}x{df.shape[1]})")
                except Exception as e:
                    logger.warning(f"  ✗ Erreur ARFF {arff_file.name}: {e}")
        except ImportError:
            logger.debug("scipy.io.arff non disponible, skip ARFF")
        
        if not self.datasets:
            logger.error("Aucun dataset chargé!")
            raise ValueError("No datasets found in data path")
        
        logger.info(f"✅ {len(self.datasets)} dataset(s) chargé(s)")
        return self.datasets
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Fusionne tous les datasets chargés.
        
        Returns:
            pd.DataFrame: Dataset fusionné et prétraité
        
        Raises:
            ValueError: Si aucun dataset n'a été chargé
        """
        if not self.datasets:
            logger.error("Aucun dataset à fusionner! Appelez load_all_datasets() d'abord")
            raise ValueError("No datasets loaded. Call load_all_datasets() first")
        
        logger.info(f"Fusion de {len(self.datasets)} dataset(s)...")
        
        # Fusionner tous les dataframes
        dfs = list(self.datasets.values())
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        logger.info(f"  Avant nettoyage: {merged_df.shape}")
        
        # Prétraitement
        merged_df = self._preprocess_raw_data(merged_df)
        
        logger.info(f"  Après nettoyage: {merged_df.shape}")
        return merged_df
    
    def _preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraitement des données brutes:
        - Conversion en numériques
        - Suppression NaN
        - Règle de complexité (Halstead)
        - Suppression features non significatives
        
        Args:
            df: Dataframe brut
        
        Returns:
            pd.DataFrame: Dataframe prétraité
        """
        original_shape = df.shape
        
        # 1. Conversion en numériques (sauf colonnes de type string pertinentes)
        logger.debug("1️⃣  Conversion numériques...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Suppression NaN / infinies
        logger.debug("2️⃣  Suppression NaN/infinies...")
        initial_rows = len(df)
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logger.info(f"   {rows_dropped} lignes NaN supprimées")
        
        # 3. Remplissage éventuels 0 (optionnel)
        # df = df.fillna(0)  # Décommenter si nécessaire
        
        # 4. Application règle de complexité (Halstead)
        if all(col in df.columns for col in self.HALSTEAD_FEATURES):
            logger.debug("3️⃣  Application règle de complexité Halstead...")
            df = self._evaluation_control(df)
        else:
            logger.warning(f"   ⚠️  Colonnes Halstead manquantes: {self.HALSTEAD_FEATURES}")
        
        # 5. Suppression features non significatives
        logger.debug("4️⃣  Suppression features non significatives...")
        cols_to_drop = [col for col in self.FEATURES_TO_DROP if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"   Colonnes supprimées: {cols_to_drop}")
        
        logger.info(f"Prétraitement complet: {original_shape} → {df.shape}")
        return df
    
    def _evaluation_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique une règle de complexité basée sur les features Halstead.
        
        Règle:
            complexityEvaluation = 1 if (n >= 300 OR v >= 1000 OR d >= 50 OR e >= 500000 OR t >= 5000)
                                  else 0
        
        Cette règle identifie les modules complexes (potentiellement bugués).
        
        Args:
            df: Dataframe avec colonnes Halstead (n, v, d, e, t)
        
        Returns:
            pd.DataFrame: Dataframe avec colonne 'complexityEvaluation'
        """
        # Vérifier présence des colonnes
        if not all(col in df.columns for col in self.HALSTEAD_FEATURES):
            logger.warning("Colonnes Halstead incomplètes, skip evaluation_control")
            return df
        
        # Appliquer règle (seuils basés sur l'approche originale inversée)
        # Si n < 300 ET v < 1000 ET ... → modèle simple → 0
        # Sinon → modèle complexe → 1
        is_simple = (
            (df['n'] < 300) & 
            (df['v'] < 1000) & 
            (df['d'] < 50) & 
            (df['e'] < 500000) & 
            (df['t'] < 5000)
        )
        
        df['complexityEvaluation'] = (~is_simple).astype(int)
        
        logger.info(f"   complexityEvaluation: {df['complexityEvaluation'].value_counts().to_dict()}")
        return df
    
    def get_target_column(self, df: pd.DataFrame) -> str:
        """
        Détecte automatiquement la colonne cible (défauts/bugs).
        
        Args:
            df: Dataframe
        
        Returns:
            str: Nom de la colonne cible
        
        Raises:
            ValueError: Si aucune colonne cible trouvée
        """
        for target in self.TARGET_COLUMNS:
            if target in df.columns:
                logger.info(f"Colonne cible détectée: {target}")
                return target
        
        # Fallback: dernière colonne
        target = df.columns[-1]
        logger.warning(f"Colonne cible non détectée, fallback sur: {target}")
        return target


def load_and_preprocess_data(data_path: str = 'data/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    FONCTION LEGACY - Maintenue pour compatibilité.
    
    Loads, concatenates, and performs initial preprocessing on the datasets.

    Args:
        data_path (str): Path to the directory containing the CSV files.

    Returns:
        tuple: (X, Y, original_data)
            - X (pd.DataFrame): Features DataFrame.
            - Y (pd.DataFrame): Target DataFrame.
            - original_data (pd.DataFrame): The full preprocessed DataFrame.
    """
    logger.info("📌 Utilisation de load_and_preprocess_data() [LEGACY]")
    logger.info("   💡 Préférez BugDataset() pour nouvelle code")
    
    dataset = BugDataset(data_path)
    dataset.load_all_datasets()
    data = dataset.merge_datasets()
    
    # Déterminer colonne cible
    target_col = dataset.get_target_column(data)
    
    # Séparer X et Y
    X = pd.DataFrame(data.drop([target_col], axis=1))
    Y = pd.DataFrame(data[[target_col]])
    
    logger.info(f"Data loaded and preprocessed. X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y, data


def evaluation_control(data: pd.DataFrame) -> pd.DataFrame:
    """
    FONCTION LEGACY - Application de règle de complexité Halstead.
    
    Applies a rule-based complexity evaluation to the data.
    Adds a 'complexityEvaluation' column.
    
    Args:
        data: DataFrame avec colonnes n, v, d, e, t
    
    Returns:
        DataFrame avec colonne 'complexityEvaluation'
    """
    logger.info("📌 Utilisation de evaluation_control() [LEGACY]")
    
    evaluation = (data.n < 300) & (data.v < 1000) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
    data['complexityEvaluation'] = (~evaluation).astype(int)
    return data


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    X_df, Y_df, full_data = load_and_preprocess_data('../../data/')
    print("\n" + "="*60)
    print("VÉRIFICATION DATASET")
    print("="*60)
    print(full_data.head())
    print("\n" + full_data.info())
    print("\nDistribution cible:")
    target_col = [col for col in full_data.columns if 'defect' in col.lower()][0]
    print(full_data[target_col].value_counts())