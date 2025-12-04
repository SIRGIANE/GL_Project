
"""
Évaluateur de modèles de prédiction de bugs
Évalue les modèles sauvegardés sur un jeu de test cohérent
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.preprocessing import load_preprocessors
from src.core.dataset import load_and_preprocess_data
from src.utils.preprocessing import (
    apply_smote, encode_labels, split_data,
    scale_features, apply_pca
)
from src.core.model import BaseModel
from src.utils.metrics import evaluate_model_performance

# Define paths
MODELS_DIR = 'bug-predictor/models/'
RESULTS_DIR = 'bug-predictor/results/'
EVALUATION_DIR = 'bug-predictor/evaluation/'
os.makedirs(EVALUATION_DIR, exist_ok=True)

class ModelEvaluator:
    """Classe pour évaluer les modèles de prédiction de bugs"""
    
    def __init__(self, models_dir: str = MODELS_DIR):
        """
        Initialise l'évaluateur
        
        Args:
            models_dir: Répertoire contenant les modèles
        """
        self.models_dir = models_dir
        self.loaded_models = {}
        self.evaluation_results = {}
        self.test_data = None
        
    def prepare_test_data(self, 
                         data_path: str = 'bug-predictor/data/',
                         use_smote: bool = True,
                         use_pca: bool = True,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données de test
        
        Args:
            data_path: Chemin vers les données
            use_smote: Appliquer SMOTE
            use_pca: Appliquer PCA
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple (X_test, y_test)
        """
        print("📊 Préparation des données de test...")
        
        # 1. Chargement des données
        X_df, Y_df, _ = load_and_preprocess_data(data_path=data_path)
        
        # 2. Application de SMOTE (optionnel)
        if use_smote:
            X_res, Y_res = apply_smote(X_df, Y_df)
        else:
            X_res, Y_res = X_df, Y_df
        
        # 3. Encodage des labels
        y_encoded, label_encoder = encode_labels(Y_res)
        
        # 4. Split des données
        X_train, X_test, y_train, y_test = split_data(
            X_res.values, y_encoded, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 5. Chargement des préprocesseurs sauvegardés
        scaler, pca = load_preprocessors(path=self.models_dir)
        
        # 6. Transformation des données de test
        X_test_scaled = scaler.transform(X_test)
        
        if use_pca and pca is not None:
            X_test_transformed = pca.transform(X_test_scaled)
            print(f"   ✓ PCA appliqué: {X_test.shape[1]} → {X_test_transformed.shape[1]} features")
        else:
            X_test_transformed = X_test_scaled
        
        self.test_data = {
            'X_test': X_test_transformed,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'scaler': scaler,
            'pca': pca
        }
        
        print(f"   ✓ Données de test prêtes: {X_test_transformed.shape[0]} échantillons")
        return X_test_transformed, y_test
    
    def load_saved_models(self, model_types: List[str] = None) -> Dict[str, BaseModel]:
        """
        Charge les modèles sauvegardés
        
        Args:
            model_types: Liste des types de modèles à charger
            
        Returns:
            Dictionnaire des modèles chargés
        """
        print("\n📂 Chargement des modèles sauvegardés...")
        
        if model_types is None:
            model_types = [
                'LogisticRegression',
                'LSTM',
                'RandomForest',
                'SVM',
                'KNN',
                'NaiveBayes',
                'DecisionTree'
            ]
        
        self.loaded_models = {}
        
        # Chercher les fichiers de modèles
        model_files = {}
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                # Extraire le nom du modèle du filename
                for model_type in model_types:
                    if model_type.lower() in filename.lower():
                        model_files[model_type] = os.path.join(self.models_dir, filename)
                        break
        
        # Charger les modèles
        for model_type, model_path in model_files.items():
            try:
                print(f"   🔍 Chargement de {model_type}...")
                
                # Charger avec joblib
                model_data = joblib.load(model_path)
                
                # Vérifier si c'est un objet BaseModel
                if isinstance(model_data, dict) and 'model' in model_data:
                    # C'est un modèle sauvegardé avec notre architecture
                    model = BaseModel("", {})
                    model.load(model_path)
                    self.loaded_models[model_type] = model
                    print(f"     ✅ {model_type} chargé ({model.model_name})")
                else:
                    # C'est un modèle scikit-learn direct
                    print(f"     ⚠️ {model_type} format ancien, création wrapper...")
                    # Créer un wrapper
                    from src.core.model_factory import ModelFactory
                    wrapper = ModelFactory.create_model(
                        model_type.lower().replace('model', ''),
                        **{}
                    )
                    wrapper.model = model_data
                    wrapper.trained = True
                    wrapper.model_name = model_type
                    self.loaded_models[model_type] = wrapper
                    
            except Exception as e:
                print(f"     ❌ Erreur chargement {model_type}: {e}")
                continue
        
        print(f"   ✓ {len(self.loaded_models)} modèles chargés")
        return self.loaded_models
    
    def evaluate_single_model(self, model: BaseModel, model_name: str) -> Dict[str, Any]:
        """
        Évalue un seul modèle
        
        Args:
            model: Modèle à évaluer
            model_name: Nom du modèle
            
        Returns:
            Dictionnaire des métriques
        """
        if self.test_data is None:
            raise ValueError("Données de test non préparées. Appelez prepare_test_data() d'abord.")
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        
        print(f"\n   📊 Évaluation de {model_name}...")
        
        try:
            # Évaluation standard
            metrics = evaluate_model_performance(
                model.model if hasattr(model, 'model') else model,
                X_test, y_test,
                model_name=model_name
            )
            
            # Si c'est un BaseModel, utiliser sa méthode evaluate
            if isinstance(model, BaseModel) and model.trained:
                model_metrics = model.evaluate(X_test, y_test)
                metrics.update(model_metrics)
            
            self.evaluation_results[model_name] = metrics
            print(f"     ✅ {model_name}: Accuracy={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"     ❌ Erreur évaluation {model_name}: {e}")
            return {}
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Évalue tous les modèles chargés
        
        Returns:
            Dictionnaire des résultats d'évaluation
        """
        print("\n🔬 Évaluation de tous les modèles...")
        
        if not self.loaded_models:
            print("⚠️ Aucun modèle chargé. Appelez load_saved_models() d'abord.")
            return {}
        
        self.evaluation_results = {}
        
        for model_name, model in self.loaded_models.items():
            self.evaluate_single_model(model, model_name)
        
        return self.evaluation_results
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Génère un rapport comparatif des modèles
        
        Returns:
            DataFrame avec les résultats comparés
        """
        if not self.evaluation_results:
            print("⚠️ Aucun résultat d'évaluation. Appelez evaluate_all_models() d'abord.")
            return pd.DataFrame()
        
        # Créer un DataFrame pour la comparaison
        comparison_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'ROC-AUC': metrics.get('roc_auc', 0),
                'Training_Time': self.loaded_models[model_name].training_time 
                    if hasattr(self.loaded_models[model_name], 'training_time') else None
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Trier par F1-Score
        df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_evaluation_results(self):
        """
        Sauvegarde les résultats d'évaluation
        """
        if not self.evaluation_results:
            print("⚠️ Aucun résultat à sauvegarder")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde des résultats bruts
        raw_results_path = f"{EVALUATION_DIR}evaluation_raw_{timestamp}.json"
        with open(raw_results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Sauvegarde du rapport comparatif
        comparison_df = self.generate_comparison_report()
        if not comparison_df.empty:
            csv_path = f"{EVALUATION_DIR}model_comparison_{timestamp}.csv"
            comparison_df.to_csv(csv_path, index=False)
            
            # Sauvegarde en format lisible
            report_path = f"{EVALUATION_DIR}evaluation_report_{timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("📊 RAPPORT D'ÉVALUATION DES MODÈLES\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Date d'évaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nombre de modèles: {len(self.evaluation_results)}\n")
                f.write(f"Échantillons de test: {self.test_data['X_test'].shape[0]}\n\n")
                
                f.write("🏆 CLASSEMENT DES MODÈLES (par F1-Score):\n")
                f.write("-" * 80 + "\n")
                
                for i, row in comparison_df.iterrows():
                    rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
                    f.write(f"{rank} {row['Model']:25s}\n")
                    f.write(f"   F1-Score:    {row['F1-Score']:.4f}\n")
                    f.write(f"   Accuracy:    {row['Accuracy']:.4f}\n")
                    f.write(f"   Precision:   {row['Precision']:.4f}\n")
                    f.write(f"   Recall:      {row['Recall']:.4f}\n")
                    f.write(f"   ROC-AUC:     {row['ROC-AUC']:.4f}\n")
                    if row['Training_Time']:
                        f.write(f"   Training:    {row['Training_Time']:.2f}s\n")
                    f.write("\n")
        
        print(f"💾 Résultats sauvegardés dans {EVALUATION_DIR}")
        return raw_results_path, csv_path
    
    def plot_model_comparison(self, save_plot: bool = True):
        """
        Génère des visualisations de comparaison des modèles
        
        Args:
            save_plot: Si True, sauvegarde les plots
        """
        if not self.evaluation_results:
            print("⚠️ Aucun résultat à visualiser")
            return
        
        comparison_df = self.generate_comparison_report()
        if comparison_df.empty:
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Comparaison des métriques principales
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
        
        # F1-Score
        ax1 = axes[0, 0]
        bars1 = ax1.barh(comparison_df['Model'], comparison_df['F1-Score'])
        ax1.set_xlabel('F1-Score')
        ax1.set_title('F1-Score par Modèle')
        ax1.invert_yaxis()
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        # Accuracy
        ax2 = axes[0, 1]
        bars2 = ax2.barh(comparison_df['Model'], comparison_df['Accuracy'])
        ax2.set_xlabel('Accuracy')
        ax2.set_title('Accuracy par Modèle')
        ax2.invert_yaxis()
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        # ROC-AUC
        ax3 = axes[1, 0]
        bars3 = ax3.barh(comparison_df['Model'], comparison_df['ROC-AUC'])
        ax3.set_xlabel('ROC-AUC')
        ax3.set_title('ROC-AUC par Modèle')
        ax3.invert_yaxis()
        for bar in bars3:
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        # Matrice des métriques
        ax4 = axes[1, 1]
        metrics_to_plot = comparison_df[['F1-Score', 'Accuracy', 'Precision', 'Recall']].T
        im = ax4.imshow(metrics_to_plot.values, aspect='auto', cmap='YlOrRd')
        ax4.set_xticks(range(len(comparison_df['Model'])))
        ax4.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax4.set_yticks(range(len(metrics_to_plot.index)))
        ax4.set_yticklabels(metrics_to_plot.index)
        ax4.set_title('Matrice des Métriques')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"{EVALUATION_DIR}model_comparison_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📈 Plot sauvegardé: {plot_path}")
        
        plt.show()
        
        # 2. Radar chart pour les meilleurs modèles
        self._plot_radar_chart(comparison_df.head(5), save_plot)
    
    def _plot_radar_chart(self, top_models_df: pd.DataFrame, save_plot: bool = True):
        """
        Génère un radar chart pour les meilleurs modèles
        
        Args:
            top_models_df: DataFrame des meilleurs modèles
            save_plot: Si True, sauvegarde le plot
        """
        if len(top_models_df) < 2:
            return
        
        # Normaliser les métriques pour le radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        normalized_data = []
        
        for metric in metrics:
            max_val = top_models_df[metric].max()
            min_val = top_models_df[metric].min()
            if max_val > min_val:
                normalized = (top_models_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized = top_models_df[metric] * 0 + 0.5  # Valeur moyenne
            normalized_data.append(normalized.values)
        
        normalized_data = np.array(normalized_data)
        
        # Créer le radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le cercle
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_models_df)))
        
        for idx, (_, row) in enumerate(top_models_df.iterrows()):
            values = normalized_data[:, idx].tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Radar Chart - Comparaison des Meilleurs Modèles', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"{EVALUATION_DIR}radar_chart_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📈 Radar chart sauvegardé: {plot_path}")
        
        plt.show()
    
    def generate_detailed_report(self) -> str:
        """
        Génère un rapport détaillé au format texte
        
        Returns:
            Rapport détaillé
        """
        if not self.evaluation_results:
            return "⚠️ Aucun résultat d'évaluation disponible."
        
        report = []
        report.append("=" * 80)
        report.append("📋 RAPPORT DÉTAILLÉ D'ÉVALUATION")
        report.append("=" * 80)
        report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Modèles évalués: {len(self.evaluation_results)}")
        
        if self.test_data:
            report.append(f"Taille du jeu de test: {self.test_data['X_test'].shape[0]} échantillons")
            report.append(f"Nombre de features: {self.test_data['X_test'].shape[1]}")
        
        report.append("\n" + "=" * 80)
        report.append("📊 RÉSULTATS PAR MODÈLE")
        report.append("=" * 80)
        
        for model_name, metrics in self.evaluation_results.items():
            report.append(f"\n🔹 {model_name}")
            report.append(f"   Accuracy:    {metrics.get('accuracy', 0):.4f}")
            report.append(f"   Precision:   {metrics.get('precision', 0):.4f}")
            report.append(f"   Recall:      {metrics.get('recall', 0):.4f}")
            report.append(f"   F1-Score:    {metrics.get('f1_score', 0):.4f}")
            report.append(f"   ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
            
            # Matrice de confusion
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                if isinstance(cm, list) and len(cm) == 2:
                    report.append(f"   Matrice de confusion:")
                    report.append(f"      [[{cm[0][0]:4d}  {cm[0][1]:4d}]")
                    report.append(f"       [{cm[1][0]:4d}  {cm[1][1]:4d}]]")
        
        # Recommandations
        report.append("\n" + "=" * 80)
        report.append("💡 RECOMMANDATIONS")
        report.append("=" * 80)
        
        # Trouver le meilleur modèle par métrique
        best_f1 = max(self.evaluation_results.items(), 
                     key=lambda x: x[1].get('f1_score', 0))
        best_accuracy = max(self.evaluation_results.items(), 
                           key=lambda x: x[1].get('accuracy', 0))
        best_roc = max(self.evaluation_results.items(), 
                      key=lambda x: x[1].get('roc_auc', 0))
        
        report.append(f"\n🎯 Meilleur modèle F1-Score: {best_f1[0]} ({best_f1[1].get('f1_score', 0):.4f})")
        report.append(f"🎯 Meilleur modèle Accuracy: {best_accuracy[0]} ({best_accuracy[1].get('accuracy', 0):.4f})")
        report.append(f"🎯 Meilleur modèle ROC-AUC: {best_roc[0]} ({best_roc[1].get('roc_auc', 0):.4f})")
        
        # Suggestions basées sur les performances
        report.append("\n📋 Suggestions:")
        
        if best_f1[1].get('f1_score', 0) > 0.8:
            report.append("   ✅ Excellentes performances! Le modèle est prêt pour la production.")
        elif best_f1[1].get('f1_score', 0) > 0.6:
            report.append("   ⚠️ Bonnes performances. Peut être amélioré avec plus de données.")
        else:
            report.append("   ❌ Performances à améliorer. Considérez:")
            report.append("       - Collecter plus de données")
            report.append("       - Rééquilibrer les classes")
            report.append("       - Essayer d'autres features")
        
        return "\n".join(report)

def evaluate_all_saved_models():
    """
    Fonction principale pour évaluer tous les modèles sauvegardés
    """
    print("🚀 Démarrage du pipeline d'évaluation...")
    
    # Créer l'évaluateur
    evaluator = ModelEvaluator()
    
    # 1. Préparer les données de test
    X_test, y_test = evaluator.prepare_test_data()
    
    # 2. Charger les modèles
    models = evaluator.load_saved_models()
    
    if not models:
        print("❌ Aucun modèle chargé. Vérifiez le répertoire des modèles.")
        return
    
    # 3. Évaluer tous les modèles
    results = evaluator.evaluate_all_models()
    
    # 4. Générer et sauvegarder les rapports
    print("\n📋 Génération des rapports...")
    
    # Rapport détaillé
    detailed_report = evaluator.generate_detailed_report()
    print(detailed_report)
    
    # Sauvegarde des résultats
    evaluator.save_evaluation_results()
    
    # Visualisations
    print("\n🎨 Génération des visualisations...")
    evaluator.plot_model_comparison(save_plot=True)
    
    # Afficher le classement
    comparison_df = evaluator.generate_comparison_report()
    if not comparison_df.empty:
        print("\n" + "=" * 80)
        print("🏆 CLASSEMENT FINAL")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ ÉVALUATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 80)
    
    return evaluator

def evaluate_specific_models(model_names: List[str]):
    """
    Évalue des modèles spécifiques
    
    Args:
        model_names: Liste des noms des modèles à évaluer
    """
    evaluator = ModelEvaluator()
    
    # Préparer les données
    evaluator.prepare_test_data()
    
    # Charger les modèles spécifiques
    evaluator.load_saved_models(model_names)
    
    # Évaluer
    results = evaluator.evaluate_all_models()
    
    # Générer rapport
    report = evaluator.generate_detailed_report()
    print(report)
    
    return evaluator

def compare_two_models(model1_path: str, model2_path: str, 
                      data_path: str = 'bug-predictor/data/'):
    """
    Compare deux modèles spécifiques
    
    Args:
        model1_path: Chemin vers le premier modèle
        model2_path: Chemin vers le second modèle
        data_path: Chemin vers les données
    """
    print("⚖️ Comparaison de deux modèles...")
    
    # Charger les modèles
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    
    # Préparer les données
    evaluator = ModelEvaluator()
    X_test, y_test = evaluator.prepare_test_data(data_path=data_path)
    
    # Évaluer chaque modèle
    results = {}
    
    for name, model in [('Model 1', model1), ('Model 2', model2)]:
        try:
            metrics = evaluate_model_performance(model, X_test, y_test, model_name=name)
            results[name] = metrics
            print(f"✅ {name}: F1={metrics.get('f1_score', 0):.3f}")
        except Exception as e:
            print(f"❌ Erreur avec {name}: {e}")
    
    # Comparaison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("📊 COMPARAISON")
        print("=" * 60)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            val1 = results.get('Model 1', {}).get(metric, 0)
            val2 = results.get('Model 2', {}).get(metric, 0)
            
            if val1 > val2:
                winner = "Model 1"
                diff = val1 - val2
            else:
                winner = "Model 2"
                diff = val2 - val1
            
            print(f"{metric:12s}: Model 1={val1:.3f} | Model 2={val2:.3f} | "
                  f"Gagnant: {winner} (+{diff:.3f})")

if __name__ == '__main__':
    """
    Exemple d'utilisation
    """
    
    # Option 1: Évaluer tous les modèles
    evaluator = evaluate_all_saved_models()
    
    # Option 2: Évaluer des modèles spécifiques
    # evaluator = evaluate_specific_models(['RandomForest', 'LogisticRegression'])
    
    # Option 3: Comparer deux modèles spécifiques
    # compare_two_models(
    #     'bug-predictor/models/RandomForestModel_20250101_120000.joblib',
    #     'bug-predictor/models/LogisticRegressionModel_20250101_120000.joblib'
    # )