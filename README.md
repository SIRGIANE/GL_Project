# Risk Predictor - Prédiction de Fichiers à Risque

## Description
Ce projet utilise l'IA pour analyser le code et prédire automatiquement les fichiers à risque dans un projet logiciel.

## Architecture
- **Backend**: Flask + ML (scikit-learn, XGBoost)
- **Frontend**: React
- **Base de données**: PostgreSQL
- **Cache**: Redis
- **Reverse Proxy**: Nginx
- **Notebooks**: Jupyter Lab
- **Orchestration**: Docker Compose

## Services
- **Backend**: http://localhost:5000
- **Frontend**: http://localhost:3000
- **Nginx**: http://localhost:80
- **Jupyter**: http://localhost:8888
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Installation et Démarrage

### Prérequis
- Docker et Docker Compose installés
- Make (optionnel, pour les commandes simplifiées)

### Configuration initiale
```bash
# Cloner le projet
git clone <your-repo>
cd GL_Project

# Configuration complète
make setup

# Ou manuellement :
cp .env.example .env
docker-compose build
```

### Démarrage

#### Mode Développement (recommandé)
```bash
make dev
# ou
docker-compose -f docker-compose.dev.yml up --build
```

#### Mode Production
```bash
make up
# ou
docker-compose up -d
```

## Commandes Utiles

```bash
make help                # Voir toutes les commandes disponibles
make dev                 # Démarrer en mode développement
make logs                # Voir les logs
make shell-backend       # Accéder au backend
make shell-db           # Accéder à PostgreSQL
make jupyter            # Lancer Jupyter Lab
make clean              # Nettoyer les conteneurs
make reset-db           # Réinitialiser la DB
```

## Fonctionnalités

### 1. Import Dataset
- Upload de fichiers de code
- Import depuis Git
- Analyse de l'historique Git

### 2. Prétraitement et Nettoyage
- Extraction des métriques de code
- Calcul de complexité
- Nettoyage des données

### 3. Entraînement Modèle ML
- Modèles : Random Forest, XGBoost, LightGBM
- Validation croisée
- Optimisation des hyperparamètres

### 4. API de Prédiction
- Endpoints RESTful
- Prédiction en temps réel
- Batch processing

### 5. Évaluation du Modèle
- Métriques : Accuracy, Precision, Recall, F1
- Validation sur données de test
- Courbes ROC

### 6. Interface Web
- Dashboard de visualisation
- Upload de fichiers
- Résultats de prédiction

## Développement

### Structure des Dossiers
```
GL_Project/
├── backend/           # API Flask + ML
├── frontend/          # Interface React
├── database/          # Scripts PostgreSQL
├── nginx/            # Configuration reverse proxy
├── notebooks/        # Jupyter pour exploration
├── data/            # Données et modèles
└── docker-compose*  # Orchestration
```

### Tests
```bash
# Tests backend
docker-compose exec backend python -m pytest

# Tests frontend
docker-compose exec frontend npm test
```

## Base de Données

### Accès PostgreSQL
```bash
make shell-db
# ou
docker-compose exec db psql -U postgres -d risk_predictor
```

### Tables Principales
- `projects`: Projets analysés
- `files`: Fichiers avec scores de risque
- `file_metrics`: Métriques détaillées
- `predictions`: Historique des prédictions

## Notebooks Jupyter

Accédez à Jupyter Lab sur http://localhost:8888 pour :
- Explorer les données
- Développer de nouveaux modèles
- Analyser les résultats

## Production

Pour déployer en production, utilisez :
```bash
docker-compose -f docker-compose.yml up -d
```

## Troubleshooting

### Problèmes courants
1. **Port occupé** : Changez les ports dans docker-compose.yml
2. **Permissions** : `sudo chown -R $USER:$USER data/`
3. **Mémoire** : Augmentez les limites Docker si nécessaire

### Logs
```bash
make logs                # Tous les services
make logs-backend       # Backend seulement
make logs-frontend      # Frontend seulement
```