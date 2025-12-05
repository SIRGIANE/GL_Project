# 🐛 Bug Predictor AI - Application Web Professionnelle

Une interface web moderne et professionnelle pour l'analyse de code Python et la prédiction de bugs, remplaçant l'interface Streamlit originale.

## ✨ Fonctionnalités

### 🎯 Interface Principale
- **Analyse en temps réel** : Interface AJAX moderne pour l'analyse de code
- **Exemples intégrés** : Code propre vs code complexe
- **Résultats instantanés** : Affichage rapide des métriques principales
- **Statut API** : Monitoring en temps réel de l'état de l'API

### 📊 Visualisations
- **Graphiques interactifs** : Chart.js pour les métriques
- **Tableaux détaillés** : 18 features avec descriptions
- **Métriques principales** : LOC, complexité, volume, difficulté
- **Export** : PNG, PDF, JSON

### 🔧 Mode Debug
- **Analyse locale** : Test des métriques sans API ML
- **Seuils visuels** : Indicateurs de dépassement des limites
- **Comparaison** : Exemples propres vs complexes

### 📱 Design Responsive
- **Mobile-first** : Compatible tous écrans
- **Bootstrap 5** : Design moderne et accessible
- **Animations** : Transitions fluides
- **Thème professionnel** : Interface claire et intuitive

## 🚀 Démarrage Rapide

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancement automatique
```bash
# Lance API + Application Web automatiquement
python start.py
```

### 3. Lancement manuel
```bash
# Terminal 1 - API Backend
python app/api.py

# Terminal 2 - Application Web
python app/web_app.py
```

### 4. Accès aux services
- **Application Web** : http://localhost:8080
- **API Backend** : http://localhost:5000

## 📁 Structure de l'Application

```
app/
├── web_app.py              # Application Flask principale
├── api.py                  # API Backend (existante)
├── templates/              # Templates HTML
│   ├── base.html          # Template de base
│   ├── index.html         # Page principale
│   ├── results.html       # Résultats détaillés
│   ├── debug.html         # Mode debug
│   ├── about.html         # À propos
│   └── error.html         # Pages d'erreur
└── static/                # Ressources statiques
    ├── css/
    │   └── style.css      # Styles personnalisés
    └── js/
        ├── main.js        # Fonctions principales
        ├── analyzer.js    # Logique d'analyse
        ├── debug.js       # Mode debug
        └── results.js     # Visualisations
```

## 🔄 Flux d'Utilisation

### Analyse Standard
1. **Saisie** : Coller le code Python dans l'éditeur
2. **Validation** : Vérification automatique de la saisie
3. **Analyse** : Envoi vers l'API pour traitement
4. **Résultats** : Affichage des métriques et prédiction
5. **Détails** : Option pour voir l'analyse complète

### Mode Debug
1. **Code de test** : Utiliser les exemples ou saisir du code
2. **Analyse locale** : Extraction des 18 métriques
3. **Seuils** : Vérification des limites critiques
4. **Visualisation** : Graphiques des features

## 🛠️ Technologies Utilisées

### Backend
- **Flask 2.3+** : Framework web Python
- **Flask-CORS** : Support CORS pour API
- **Radon** : Analyse statique de code
- **Scikit-learn** : Machine Learning

### Frontend
- **Bootstrap 5.3** : Framework CSS
- **Chart.js** : Graphiques interactifs
- **Font Awesome 6** : Icônes
- **Vanilla JavaScript** : Interactivité

### Infrastructure
- **HTML5/CSS3** : Structure et style
- **Jinja2** : Moteur de templates
- **AJAX/Fetch** : Communication asynchrone

## 📡 API Endpoints

### Application Web
- `GET /` : Page principale
- `POST /analyze` : Analyse avec redirection
- `GET /debug` : Page de debug
- `GET /about` : Page à propos

### API JSON
- `POST /api/analyze` : Analyse AJAX
- `POST /api/debug` : Debug local
- `GET /api/status` : Statut des services
- `GET /api/health` : Santé de l'API

## 🎨 Personnalisation

### Thème et Couleurs
Modifiez `static/css/style.css` pour personnaliser :
- Variables CSS (`:root`)
- Couleurs principales
- Animations et transitions
- Responsive design

### Exemples de Code
Modifiez `static/js/analyzer.js` et `templates/debug.html` pour :
- Ajouter de nouveaux exemples
- Personnaliser les snippets
- Modifier les descriptions

## 🔍 Comparaison Streamlit vs Flask

| Aspect | Streamlit | Flask Web App |
|--------|-----------|---------------|
| **Performance** | Rechargement complet | AJAX rapide |
| **UX/UI** | Basique | Professionnel |
| **Personnalisation** | Limitée | Totale liberté |
| **Responsive** | Basique | Optimisé mobile |
| **Déploiement** | Simple | Production-ready |
| **Intégration** | API séparée | Architecture unifiée |

## 📊 Métriques Analysées

### Métriques de Base (Radon)
- **LOC** : Lignes de code totales
- **SLOC** : Lignes de code source
- **LLOC** : Lignes logiques
- **Commentaires** : Lignes de documentation
- **Complexité** : Cyclomatique de McCabe

### Métriques Halstead
- **Volume (v)** : Taille du programme
- **Difficulté (d)** : Complexité de compréhension
- **Effort** : Effort de développement estimé
- **Temps** : Temps de développement
- **Opérateurs/Opérandes** : Éléments du programme

### Seuils Critiques
Code considéré comme potentiellement bogué si :
- `n ≥ 300` (Longueur Halstead)
- `v ≥ 1000` (Volume)
- `d ≥ 50` (Difficulté)
- `effort ≥ 500000` (Effort)
- `time ≥ 5000` (Temps)

## 🚀 Déploiement Production

### Configuration
```python
# web_app.py - Mode production
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Docker (optionnel)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "app/web_app.py"]
```

### Nginx (reverse proxy)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🐛 Dépannage

### API non disponible
```bash
# Vérifier le port 5000
lsof -i :5000

# Relancer l'API
python app/api.py
```

### Application web ne démarre pas
```bash
# Vérifier les dépendances
pip install flask flask-cors

# Vérifier le port 8080
lsof -i :8080
```

### Erreurs JavaScript
- Ouvrir les outils développeur (F12)
- Vérifier la console pour les erreurs
- Vérifier que les fichiers statiques sont accessibles

## 📝 Logs et Monitoring

### Logs Application
```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Monitoring API
- Status : `GET /api/status`
- Health : `GET /api/health`
- Métriques en temps réel dans l'interface

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commiter les changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

**🎉 Votre application web professionnelle est maintenant prête !**

Lancez `python start.py` et accédez à http://localhost:8080 pour découvrir l'interface moderne de Bug Predictor AI.