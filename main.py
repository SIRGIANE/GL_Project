#!/usr/bin/env python3
"""
BUG PREDICTOR - Pipeline complet automatique
Fusion des datasets NASA → Prétraitement → Entraînement 5 modèles → Évaluation → Production
Exécutez: python main.py [--test-size 0.2] [--cv-folds 5] [--n-jobs -1]
"""

import sys
import os
import logging
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# FIX: Supporter Unicode sur Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Logger global avec support Unicode
logger = logging.getLogger("bug_predictor")
logger.setLevel(logging.DEBUG)

# Handler console avec encodage UTF-8
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.stream.reconfigure(encoding='utf-8') if hasattr(console_handler.stream, 'reconfigure') else None
console_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# ============================================================================
# IMPORTATIONS MODULES LOCAUX
# ============================================================================

try:
    from src.core.dataset import BugDataset
    from src.utils.preprocessing import BugPreprocessor
    from src.core.model_factory import ModelFactory
    from src.pipeline.trainer import ModelTrainer
    logger.debug("[OK] Modules locaux importes avec succes")
except ImportError as e:
    logger.warning(f"[WARN] Modules locaux non trouves: {e}")
    BugDataset = None
    BugPreprocessor = None
    ModelFactory = None
    ModelTrainer = None

# ============================================================================
# VALIDATION ARGUMENTS
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse et valide les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="BUG PREDICTOR - Pipeline ML complet pour prediction de bugs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py                              # Configuration par defaut
  python main.py --test-size 0.25 --n-jobs 4 # 25% test, 4 cores
  python main.py --cv-folds 10                # Validation croisee 10-fold
        """
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion de test (defaut: 0.2)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Nombre de folds pour validation croisee (defaut: 5)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Nombre de jobs paralleles (-1 = tous les cores; defaut: -1)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine aleatoire (defaut: 42)"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=20,
        help="Nombre de composantes PCA (defaut: 20)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbose (DEBUG)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not (0 < args.test_size < 1):
        raise ValueError(f"test-size doit etre entre 0 et 1, recu: {args.test_size}")
    if args.cv_folds < 2:
        raise ValueError(f"cv-folds doit etre >= 2, recu: {args.cv_folds}")
    if args.pca_components < 1:
        raise ValueError(f"pca-components doit etre >= 1, recu: {args.pca_components}")
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    logger.debug(f"Arguments: test_size={args.test_size}, cv_folds={args.cv_folds}, n_jobs={args.n_jobs}")
    
    return args


# ============================================================================
# ÉTAPE 1: VÉRIFICATION & CRÉATION DOSSIERS
# ============================================================================

def setup_directories() -> Tuple[Path, Path, Path, Path]:
    """Crée les dossiers nécessaires et retourne les chemins"""
    logger.info("[1/8] Verification et creation des dossiers...")
    
    required_dirs = {
        'data': PROJECT_ROOT / 'data',
        'models': PROJECT_ROOT / 'models',
        'output': PROJECT_ROOT / 'output',
        'logs': PROJECT_ROOT / 'logs'
    }
    
    for dir_name, dir_path in required_dirs.items():
        dir_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"   [OK] {dir_name}/ existe")
    
    # Vérifier fichiers data
    data_files = list(required_dirs['data'].glob('*'))
    if not data_files:
        logger.error("   [FAIL] Le dossier data/ est vide!")
        logger.info("   [INFO] Placez vos fichiers .arff ou .csv dans data/")
        sys.exit(1)
    
    logger.info(f"   [INFO] {len(data_files)} fichier(s) trouve(s) dans data/")
    for f in data_files[:3]:
        logger.debug(f"      - {f.name}")
    if len(data_files) > 3:
        logger.debug(f"      - ... et {len(data_files) - 3} autre(s)")
    
    # Créer dossiers de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = required_dirs['output'] / timestamp
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    
    for directory in [output_dir, models_dir, metrics_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Fichier log
    log_file = required_dirs['logs'] / f"run_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"   [OK] Logs ecrits dans: {log_file.name}")
    logger.info(f"   [OK] Resultats dans: output/{timestamp}/")
    
    return models_dir, metrics_dir, plots_dir, required_dirs['data']


# ============================================================================
# ÉTAPE 2: CHARGEMENT & FUSION DATASETS
# ============================================================================

def load_and_merge_datasets(data_dir: Path) -> pd.DataFrame:
    """Charge et fusionne tous les datasets (ARFF/CSV)"""
    logger.info("[2/8] Chargement des donnees...")
    
    try:
        if BugDataset is not None:
            # Utiliser le module custom si disponible
            dataset = BugDataset(str(data_dir))
            datasets = dataset.load_all_datasets()
            
            if not datasets:
                logger.error("   [FAIL] Aucun dataset charge avec BugDataset!")
                sys.exit(1)
            
            logger.info(f"   [OK] {len(datasets)} dataset(s) charge(s)")
            merged_df = dataset.merge_datasets()
        else:
            # Fallback: charger manuellement CSV/ARFF
            logger.warning("   [WARN] BugDataset non disponible, chargement manuel...")
            dfs = []
            
            # Charger CSV
            for csv_file in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                    logger.debug(f"      - CSV: {csv_file.name} ({df.shape})")
                except Exception as e:
                    logger.warning(f"      [FAIL] Erreur CSV {csv_file.name}: {e}")
            
            # Charger ARFF si possible
            try:
                from scipy.io import arff
                for arff_file in data_dir.glob("*.arff"):
                    try:
                        data, meta = arff.loadarff(arff_file)
                        df = pd.DataFrame(data)
                        dfs.append(df)
                        logger.debug(f"      - ARFF: {arff_file.name} ({df.shape})")
                    except Exception as e:
                        logger.warning(f"      [FAIL] Erreur ARFF {arff_file.name}: {e}")
            except ImportError:
                logger.warning("   [WARN] scipy non disponible, skip ARFF")
            
            if not dfs:
                logger.error("   [FAIL] Aucun fichier charge!")
                sys.exit(1)
            
            merged_df = pd.concat(dfs, axis=0, ignore_index=True)
            logger.info(f"   [OK] {len(dfs)} fichier(s) fusionne(s)")
    
    except Exception as e:
        logger.error(f"   [FAIL] Erreur chargement datasets: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"   [INFO] Dataset fusionne: {merged_df.shape[0]} lignes, {merged_df.shape[1]} colonnes")
    logger.debug(f"      Colonnes: {', '.join(str(c) for c in merged_df.columns[:5])}...")
    
    return merged_df


# ============================================================================
# ÉTAPE 3: PRÉTRAITEMENT
# ============================================================================

def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    pca_components: int = 20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any, Any, str]:
    """Prétraitement complet: encodage, SMOTE, split, scale, PCA"""
    logger.info("[3/8] Pretraitement des donnees...")
    
    # 3.1 Détection colonne cible
    target_columns = [col for col in df.columns 
                     if any(x in col.lower() for x in ['bug', 'defect', 'bug_', 'defect_'])]
    
    if target_columns:
        target_col = target_columns[0]
    else:
        target_col = df.columns[-1]
    
    logger.info(f"   [OK] Colonne cible: {target_col}")
    
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    # 3.1.1 CONVERSION TYPE Y - FIX PRINCIPAL
    logger.debug(f"   [DEBUG] Type y avant conversion: {y.dtype}, uniques: {y.unique()[:5]}")
    
    # Convertir tous les éléments en int ou string uniforme
    y = y.astype(str).str.lower().str.strip()
    y = y.map({'0': 0, 'false': 0, 'no': 0, '1': 1, 'true': 1, 'yes': 1})
    
    # Si la conversion échoue, essayer numérique
    if y.isnull().any():
        logger.warning(f"   [WARN] Conversion partielle, remplissage des NaN")
        y = y.fillna(0).astype(int)
    else:
        y = y.astype(int)
    
    logger.debug(f"   [DEBUG] Type y apres conversion: {y.dtype}, uniques: {y.unique()}")
    
    # 3.2 Gestion NaN dans X
    if X.isnull().sum().sum() > 0:
        nan_count = X.isnull().sum().sum()
        logger.warning(f"   [WARN] {nan_count} NaN detectes, remplissage a 0")
        X = X.fillna(0)
    
    # 3.3 Conversion X en numériques
    logger.debug("   [DEBUG] Conversion X en numeriques...")
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # 3.4 SMOTE (rééquilibrage)
    logger.info(f"   [INFO] Avant SMOTE: {dict(pd.Series(y).value_counts())}")
    
    try:
        smote = SMOTE(random_state=random_state, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"   [INFO] Apres SMOTE: {dict(pd.Series(y_res).value_counts())}")
    except Exception as e:
        logger.warning(f"   [WARN] SMOTE echoue: {e}, skip")
        X_res, y_res = X, y
    
    # 3.5 Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res,
        test_size=test_size,
        random_state=random_state,
        stratify=y_res
    )
    
    logger.info(f"   [OK] Split: Train {X_train.shape} | Test {X_test.shape}")
    
    # 3.6 Normalisation (fit sur train, apply sur test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"   [OK] Features normalisees (StandardScaler)")
    
    # 3.7 PCA (fit sur train, apply sur test)
    n_features = min(pca_components, X_train.shape[1])
    pca = PCA(n_components=n_features, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"   [OK] PCA: {X_train.shape[1]} -> {n_features} features")
    logger.info(f"   [OK] Variance expliquee: {explained_var:.1%}")
    
    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca, None, target_col


# ============================================================================
# ÉTAPE 4: ENTRAÎNEMENT MODÈLES (5 modèles)
# ============================================================================

def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1,
    random_state: int = 42
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    """Entraîne 5 modèles et évalue"""
    logger.info("[4/8] Entrainement de 5 modeles...")
    
    models_config = [
        {
            'name': 'Random Forest',
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': random_state,
                'n_jobs': n_jobs
            }
        },
        {
            'name': 'Logistic Regression',
            'class': LogisticRegression,
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': random_state,
                'n_jobs': n_jobs,
                'solver': 'lbfgs'
            }
        },
        {
            'name': 'SVM (RBF)',
            'class': SVC,
            'params': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': random_state
            }
        },
        {
            'name': 'K-Nearest Neighbors',
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': 5,
                'n_jobs': n_jobs
            }
        },
        {
            'name': 'Gradient Boosting',
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': random_state
            }
        }
    ]
    
    trained_models = {}
    metrics_results = []
    
    for config in models_config:
        try:
            model_name = config['name']
            logger.info(f"   [{model_name}] Entrainement...")
            
            # Créer et entraîner
            model = config['class'](**config['params'])
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Métriques
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = 0.5
            
            # Validation croisée
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, random_state=random_state, shuffle=True),
                scoring='f1_weighted'
            )
            
            metrics = {
                'Model': model_name,
                'Accuracy': acc,
                'F1-Score': f1,
                'Precision': prec,
                'Recall': rec,
                'ROC-AUC': roc_auc,
                'CV-Mean': cv_scores.mean(),
                'CV-Std': cv_scores.std()
            }
            
            metrics_results.append(metrics)
            trained_models[model_name] = model
            
            logger.info(f"   [OK] F1={f1:.3f} | CV={cv_scores.mean():.3f}+/-{cv_scores.std():.3f}")
            
        except Exception as e:
            logger.error(f"   [FAIL] Erreur {model_name}: {e}", exc_info=True)
            continue
    
    if not trained_models:
        logger.error("   [FAIL] Aucun modele entraine avec succes!")
        sys.exit(1)
    
    logger.info(f"   [OK] {len(trained_models)} modele(s) entraine(s)")
    
    return trained_models, metrics_results


# ============================================================================
# ÉTAPE 5: ÉVALUATION & CLASSEMENT
# ============================================================================

def evaluate_models(
    trained_models: Dict[str, Any],
    metrics_results: List[Dict[str, float]]
) -> Tuple[pd.DataFrame, str, Any]:
    """Évalue et classe les modèles"""
    logger.info("[5/8] Evaluation des modeles...")
    
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    logger.info("\n   CLASSEMENT:")
    logger.info("   " + "-" * 70)
    
    for i, row in metrics_df.iterrows():
        medal = "[1]" if i == 0 else "[2]" if i == 1 else "[3]" if i == 2 else f"[{i+1}]"
        logger.info(
            f"   {medal} {row['Model']:25s} | F1: {row['F1-Score']:.4f} | "
            f"CV: {row['CV-Mean']:.4f}+/-{row['CV-Std']:.4f}"
        )
    
    best_row = metrics_df.iloc[0]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    
    logger.info(f"\n   Meilleur modele: {best_model_name}")
    logger.info(f"   Score F1: {best_row['F1-Score']:.4f}")
    
    return metrics_df, best_model_name, best_model


# ============================================================================
# ÉTAPE 6: VISUALISATIONS
# ============================================================================

def generate_visualizations(
    metrics_df: pd.DataFrame,
    trained_models: Dict[str, Any],
    best_model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plots_dir: Path
) -> None:
    """Génère graphiques et matrices de confusion"""
    logger.info("[6/8] Generation des visualisations...")
    
    try:
        # 6.1 Comparaison modèles
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.18
        
        metrics_to_plot = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            values = metrics_df[metric].values
            ax.bar(x + i*width - width*1.5, values, width, label=metric, color=color, alpha=0.8)
        
        ax.set_xlabel('Modeles', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Comparaison des Performances (5 Modeles)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Model'].values, rotation=30, ha='right', fontsize=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plot_path = plots_dir / "01_model_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   [OK] Graphique: 01_model_comparison.png")
        
        # 6.2 Matrice de confusion (meilleur modèle)
        best_model = trained_models[best_model_name]
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Bug', 'Bug'],
            yticklabels=['No Bug', 'Bug'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_title(f'Matrice de Confusion - {best_model_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Verite', fontsize=11, fontweight='bold')
        ax.set_xlabel('Prediction', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        cm_path = plots_dir / "02_confusion_matrix_best_model.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   [OK] Matrice confusion: 02_confusion_matrix_best_model.png")
        
    except Exception as e:
        logger.error(f"   [WARN] Erreur visualisation: {e}", exc_info=True)


# ============================================================================
# ÉTAPE 7: SAUVEGARDE ARTEFACTS
# ============================================================================

def save_artifacts(
    trained_models: Dict[str, Any],
    metrics_df: pd.DataFrame,
    best_model_name: str,
    best_model: Any,
    scaler: StandardScaler,
    pca: PCA,
    label_encoder: Any,
    target_col: str,
    models_dir: Path,
    metrics_dir: Path,
    df_original: pd.DataFrame
) -> None:
    """Sauvegarde modèles, métriques, encodeurs, PCA"""
    logger.info("[7/8] Sauvegarde des artefacts...")
    
    try:
        # 7.1 Meilleur modèle
        best_model_path = models_dir / "best_model.joblib"
        joblib.dump(best_model, best_model_path)
        logger.info(f"   [OK] Meilleur modele: best_model.joblib ({best_model_path.stat().st_size / 1024:.1f} KB)")
        
        # 7.2 Tous les modèles
        for model_name, model in trained_models.items():
            safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            model_path = models_dir / f"{safe_name}_model.joblib"
            joblib.dump(model, model_path)
        logger.info(f"   [OK] {len(trained_models)} modele(s) individuels sauvegardes")
        
        # 7.3 Préprocesseurs
        joblib.dump(scaler, models_dir / "scaler.joblib")
        joblib.dump(pca, models_dir / "pca.joblib")
        logger.info(f"   [OK] Preprocesseurs: scaler.joblib, pca.joblib")
        
        # 7.4 Métriques CSV et JSON
        metrics_csv = metrics_dir / "model_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"   [OK] Metriques CSV: model_metrics.csv")
        
        metrics_json = metrics_dir / "model_metrics.json"
        metrics_df.to_json(metrics_json, orient='records', indent=2)
        logger.info(f"   [OK] Metriques JSON: model_metrics.json")
        
        # 7.5 Configuration
        config = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': df_original.shape[0],
                'total_features': df_original.shape[1] - 1,
                'target_column': target_col,
                'class_distribution': df_original[target_col].value_counts().to_dict()
            },
            'preprocessing': {
                'scaler': 'StandardScaler',
                'pca_components': pca.n_components_,
                'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
                'smote_applied': True
            },
            'models_trained': list(trained_models.keys()),
            'best_model': best_model_name,
            'best_f1_score': float(metrics_df.iloc[0]['F1-Score']),
            'best_accuracy': float(metrics_df.iloc[0]['Accuracy'])
        }
        
        config_path = metrics_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"   [OK] Configuration: config.json")
        
    except Exception as e:
        logger.error(f"   [FAIL] Erreur sauvegarde: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# ÉTAPE 8: RAPPORT FINAL & USAGE
# ============================================================================

def print_summary_report(
    metrics_df: pd.DataFrame,
    best_model_name: str,
    best_row: pd.Series,
    trained_models: Dict[str, Any],
    models_dir: Path,
    metrics_dir: Path,
    plots_dir: Path,
    output_dir: Path
) -> None:
    """Affiche rapport final et instructions"""
    logger.info("[8/8] Rapport final...")
    
    logger.info("\n" + "=" * 80)
    logger.info("[SUCCESS] PIPELINE COMPLETE AVEC SUCCES!")
    logger.info("=" * 80)
    
    logger.info(f"\nRESULTATS:")
    logger.info(f"   - Modeles entraines: {len(trained_models)}")
    logger.info(f"   - Meilleur modele: {best_model_name}")
    logger.info(f"   - Score F1: {best_row['F1-Score']:.4f}")
    logger.info(f"   - Accuracy: {best_row['Accuracy']:.4f}")
    logger.info(f"   - ROC-AUC: {best_row['ROC-AUC']:.4f}")
    
    logger.info(f"\nFICHIERS GENERES:")
    logger.info(f"   [models/]        ({len(list(models_dir.glob('*.joblib')))} fichiers)")
    logger.info(f"      - best_model.joblib")
    logger.info(f"      - *.joblib (modeles individuels)")
    
    logger.info(f"   [metrics/]       ({len(list(metrics_dir.glob('*')))} fichiers)")
    logger.info(f"      - model_metrics.csv")
    logger.info(f"      - model_metrics.json")
    logger.info(f"      - config.json")
    
    logger.info(f"   [plots/]         ({len(list(plots_dir.glob('*.png')))} graphiques)")
    for plot_file in sorted(plots_dir.glob("*.png"))[:5]:
        logger.info(f"      - {plot_file.name}")
    
    logger.info(f"\nPROCHAINES ETAPES:")
    logger.info(f"   1. Voir les metriques: cat {metrics_dir / 'model_metrics.csv'}")
    logger.info(f"   2. Voir le graphique: output/{output_dir.name}/plots/01_model_comparison.png")
    logger.info(f"   3. Charger le modele: joblib.load('{models_dir}/best_model.joblib')")
    
    logger.info("\n" + "=" * 80 + "\n")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Pipeline complet d'entraînement"""
    try:
        # Affiche bannière
        logger.info("\n" + "=" * 80)
        logger.info("[START] BUG PREDICTOR - PIPELINE ML COMPLET")
        logger.info("=" * 80)
        logger.info(f"[TIME] Debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Parse arguments
        args = parse_arguments()
        
        # [1] Dossiers
        models_dir, metrics_dir, plots_dir, data_dir = setup_directories()
        
        # [2] Charger & fusionner données
        df_original = load_and_merge_datasets(data_dir)
        
        # [3] Prétraitement
        X_train, X_test, y_train, y_test, scaler, pca, label_encoder, target_col = preprocess_data(
            df_original,
            test_size=args.test_size,
            pca_components=args.pca_components,
            random_state=args.random_state
        )
        
        # [4] Entraîner 5 modèles
        trained_models, metrics_results = train_models(
            X_train, X_test, y_train, y_test,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
            random_state=args.random_state
        )
        
        # [5] Évaluer
        metrics_df, best_model_name, best_model = evaluate_models(trained_models, metrics_results)
        
        # [6] Visualisations
        generate_visualizations(
            metrics_df, trained_models, best_model_name,
            X_test, y_test, plots_dir
        )
        
        # [7] Sauvegarder
        save_artifacts(
            trained_models, metrics_df, best_model_name, best_model,
            scaler, pca, label_encoder, target_col,
            models_dir, metrics_dir, df_original
        )
        
        # [8] Rapport
        output_dir = models_dir.parent
        best_row = metrics_df.iloc[0]
        print_summary_report(
            metrics_df, best_model_name, best_row, trained_models,
            models_dir, metrics_dir, plots_dir, output_dir
        )
        
        logger.info("[DONE] Pipeline complte avec succes!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\n[CANCEL] Pipeline interrompu par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"\n\n[ERROR] Erreur fatale: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)