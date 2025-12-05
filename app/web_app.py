"""
Application Web Flask Professionnelle - Bug Predictor AI
Remplace l'interface Streamlit par une interface web moderne
"""
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_cors import CORS
import requests
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import traceback
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import hashlib
import uuid

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'bug-predictor-secret-key-2025-advanced'
CORS(app)

# Configuration avancée
API_URL = "http://localhost:5001/api"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / 'temp_uploads'
ALLOWED_EXTENSIONS = {'py', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max
CACHE_DURATION = timedelta(hours=1)
RESULTS_HISTORY_LIMIT = 50

# Créer les dossiers nécessaires
UPLOAD_FOLDER.mkdir(exist_ok=True)
(PROJECT_ROOT / 'temp_cache').mkdir(exist_ok=True)
(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)  # Créer le dossier logs AVANT la configuration logging

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from radon.raw import analyze
    from radon.metrics import h_visit
    from radon.complexity import cc_visit
    HAS_RADON = True
    print("✓ Dépendances d'analyse chargées avec succès")
except Exception as e:
    print(f"⚠️ Erreur d'import des dépendances: {e}")
    HAS_RADON = False

# Logging setup avancé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'web_app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Créer le dossier de logs
(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)


class APIClient:
    """Client avancé pour communiquer avec l'API de prédiction"""
    def __init__(self, base_url=API_URL):
        self.base = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        self.cache = {}
        
    def health(self):
        """Vérification de l'état de l'API avec cache"""
        cache_key = 'health_check'
        if cache_key in self.cache:
            cached_time, result = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return result
        
        try:
            r = self.session.get(f"{self.base}/health", timeout=5)
            result = r.status_code == 200
            self.cache[cache_key] = (datetime.now(), result)
            return result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def status(self):
        """Informations détaillées sur le statut"""
        try:
            r = self.session.get(f"{self.base}/status", timeout=10)
            return r.json() if r.ok else {}
        except Exception as e:
            logger.error(f"Status error: {e}")
            return {}

    def analyze_code(self, code, analysis_id=None):
        """Analyse du code avec gestion d'erreurs avancée et cache"""
        # Générer un hash du code pour le cache
        code_hash = hashlib.md5(code.encode()).hexdigest()
        cache_key = f"analysis_{code_hash}"
        
        # Vérifier le cache
        if cache_key in self.cache:
            cached_time, result = self.cache[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                logger.info(f"Retour du cache pour l'analyse {code_hash[:8]}")
                return result
        
        try:
            payload = {
                "code": code,
                "analysis_id": analysis_id or str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "client": "web_app"
            }
            
            r = self.session.post(
                f"{self.base}/analyze/code",
                json=payload,
                timeout=60
            )
            
            if r.ok:
                result = r.json()
                # Mettre en cache le résultat
                self.cache[cache_key] = (datetime.now(), result)
                logger.info(f"Analyse réussie pour {code_hash[:8]}")
                return result
            else:
                error_msg = r.text if r.text else f"HTTP {r.status_code}"
                return {"error": error_msg}
                
        except requests.exceptions.Timeout:
            return {"error": "Timeout - L'analyse prend trop de temps"}
        except requests.exceptions.ConnectionError:
            return {"error": "Impossible de se connecter à l'API"}
        except Exception as e:
            logger.error(f"Analyze error: {e}")
            return {"error": str(e)}

    def get_models_info(self):
        """Informations sur les modèles disponibles"""
        try:
            r = self.session.get(f"{self.base}/models", timeout=10)
            return r.json() if r.ok else {}
        except Exception as e:
            logger.error(f"Models info error: {e}")
            return {}


def extract_local_features(code_str):
    """Extraction locale des métriques pour debug et validation"""
    if not HAS_RADON:
        return None, None
    
    try:
        # Validation du code
        if not code_str or not code_str.strip():
            raise ValueError("Code vide fourni")
        
        # Vérification syntaxique basique
        try:
            compile(code_str, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning(f"Code avec erreur de syntaxe: {e}")
            # Continuer quand même pour l'analyse des métriques
        
        # Analyse basique avec Radon
        raw = analyze(code_str)
        f1 = raw.loc
        f13 = raw.sloc
        f14 = raw.comments
        f15 = raw.blank
        f16 = raw.lloc

        # Métriques de Halstead
        ast = h_visit(code_str)
        hal = ast.total
        f17 = hal.h1
        f18 = hal.h2
        f19 = hal.N1
        f20 = hal.N2
        f5 = hal.length
        f6 = hal.volume
        f7 = (f6 / f5) if f5 > 0 else 0
        f8 = hal.difficulty
        f9 = f7 * f6
        f10 = hal.effort
        f12 = hal.time

        # Complexité cyclomatique
        cc = cc_visit(code_str)
        try:
            f2 = cc[-1].complexity if cc else 0
        except Exception:
            f2 = 0

        # Évaluation de complexité
        f21 = int(((f5 >= 300) or (f6 >= 1000) or (f8 >= 50) or (f10 >= 500000) or (f12 >= 5000)))

        features = [f1, f2, f5, f6, f7, f8, f9, f10, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]
        
        # Calculs additionnels
        maintainability_index = max(0, (171 - 5.2 * np.log(f6) - 0.23 * f2 - 16.2 * np.log(f1)) * 100 / 171) if f6 > 0 and f1 > 0 else 0
        
        metrics = {
            "LOC": f1, "cyclomatic_complexity": f2, "halstead_length": f5,
            "halstead_volume": f6, "halstead_level": f7, "halstead_difficulty": f8,
            "halstead_intelligence": f9, "halstead_effort": f10, "halstead_time": f12,
            "source_lines": f13, "comment_lines": f14, "blank_lines": f15,
            "logical_lines": f16, "unique_operators": f17, "unique_operands": f18,
            "total_operators": f19, "total_operands": f20, "complexity_evaluation": f21,
            "maintainability_index": round(maintainability_index, 2),
            "comments_ratio": round((f14 / f1) * 100, 2) if f1 > 0 else 0,
            "blank_ratio": round((f15 / f1) * 100, 2) if f1 > 0 else 0
        }
        
        return features, metrics
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None, None


def allowed_file(filename):
    """Vérification des extensions de fichiers autorisées"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_code_input(code):
    """Validation avancée du code d'entrée"""
    if not code or not code.strip():
        return False, "Code vide"
    
    if len(code) > MAX_FILE_SIZE:
        return False, f"Code trop volumineux (max {MAX_FILE_SIZE} bytes)"
    
    # Vérifications de sécurité basiques
    dangerous_patterns = ['import os', 'import subprocess', '__import__', 'eval(', 'exec(']
    for pattern in dangerous_patterns:
        if pattern in code.lower():
            logger.warning(f"Code potentiellement dangereux détecté: {pattern}")
    
    return True, "Code valide"


def save_analysis_to_history(code, result, user_id=None):
    """Sauvegarder l'analyse dans l'historique"""
    try:
        history_file = PROJECT_ROOT / 'temp_cache' / 'analysis_history.json'
        history = []
        
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # Limiter l'historique
        if len(history) >= RESULTS_HISTORY_LIMIT:
            history = history[-(RESULTS_HISTORY_LIMIT-1):]
        
        history.append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "code_hash": hashlib.md5(code.encode()).hexdigest(),
            "code_snippet": code[:200] + "..." if len(code) > 200 else code,
            "result": result,
            "user_id": user_id or session.get('user_id', 'anonymous')
        })
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Erreur sauvegarde historique: {e}")


def get_analysis_history(limit=10):
    """Récupérer l'historique des analyses"""
    try:
        history_file = PROJECT_ROOT / 'temp_cache' / 'analysis_history.json'
        if not history_file.exists():
            return []
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        return history[-limit:] if history else []
    except Exception as e:
        logger.error(f"Erreur lecture historique: {e}")
        return []


# Initialize API client
client = APIClient()

# Générer un ID de session unique
@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Log des requêtes
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.route('/')
def index():
    """Page principale - Prédiction de bugs"""
    api_status = client.health()
    status_info = client.status() if api_status else {}
    models_info = client.get_models_info() if api_status else {}
    recent_analyses = get_analysis_history(5)
    
    return render_template('index.html', 
                         api_status=api_status, 
                         status_info=status_info,
                         models_info=models_info,
                         recent_analyses=recent_analyses)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyse du code et prédiction avec gestion avancée"""
    analysis_id = str(uuid.uuid4())
    
    try:
        # Récupération du code depuis form ou fichier
        code = None
        source = "manual"
        
        if 'code' in request.form and request.form['code'].strip():
            code = request.form['code'].strip()
            source = "form"
        elif 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                try:
                    code = file.read().decode('utf-8')
                    source = f"file:{secure_filename(file.filename)}"
                except UnicodeDecodeError:
                    flash('Erreur: Impossible de décoder le fichier (encodage non supporté)', 'error')
                    return redirect(url_for('index'))
            else:
                flash('Fichier non valide (extensions autorisées: .py, .txt)', 'warning')
                return redirect(url_for('index'))
        
        if not code:
            flash('Veuillez fournir du code à analyser', 'warning')
            return redirect(url_for('index'))
        
        # Validation du code
        is_valid, validation_msg = validate_code_input(code)
        if not is_valid:
            flash(f'Code non valide: {validation_msg}', 'error')
            return redirect(url_for('index'))
        
        logger.info(f"Analyse démarrée - ID: {analysis_id}, Source: {source}, Taille: {len(code)} chars")
        
        # Analyse via API
        response = client.analyze_code(code, analysis_id)
        
        if "error" in response:
            error_msg = response["error"]
            logger.error(f"Erreur d'analyse {analysis_id}: {error_msg}")
            flash(f'Erreur d\'analyse: {error_msg}', 'error')
            return redirect(url_for('index'))
        
        # Extraction locale pour validation
        local_features, local_metrics = extract_local_features(code)
        
        # Formatage des résultats
        result = {
            'analysis_id': analysis_id,
            'code': code,
            'source': source,
            'is_bug': response.get('is_bug', False),
            'probability': response.get('probability', 0.0),
            'risk_level': response.get('risk_level', 'N/A'),
            'model_used': response.get('model_used', 'unknown'),
            'features': response.get('features', []),
            'metrics': response.get('metrics', {}),
            'local_metrics': local_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': response.get('processing_time', 'N/A')
        }
        
        # Sauvegarder dans l'historique
        save_analysis_to_history(code, result, session.get('user_id'))
        
        logger.info(f"Analyse terminée - ID: {analysis_id}, Prédiction: {result['is_bug']}, "
                   f"Probabilité: {result['probability']:.3f}")
        
        return render_template('results.html', result=result)
    
    except Exception as e:
        logger.error(f"Erreur inattendue dans l'analyse {analysis_id}: {e}")
        logger.error(traceback.format_exc())
        flash('Une erreur inattendue s\'est produite lors de l\'analyse', 'error')
        return redirect(url_for('index'))


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint pour analyse AJAX avec validation avancée"""
    analysis_id = str(uuid.uuid4())
    
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'Code requis', 'analysis_id': analysis_id}), 400
        
        code = data['code'].strip()
        if not code:
            return jsonify({'error': 'Code non vide requis', 'analysis_id': analysis_id}), 400
        
        # Validation
        is_valid, validation_msg = validate_code_input(code)
        if not is_valid:
            return jsonify({'error': f'Validation échouée: {validation_msg}', 'analysis_id': analysis_id}), 400
        
        # Options d'analyse
        options = data.get('options', {})
        include_local_metrics = options.get('include_local_metrics', True)
        
        start_time = datetime.now()
        
        # Analyse via API
        response = client.analyze_code(code, analysis_id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if "error" in response:
            return jsonify({
                'error': response['error'],
                'analysis_id': analysis_id,
                'processing_time': processing_time
            }), 500
        
        # Ajouter métriques locales si demandées
        if include_local_metrics:
            local_features, local_metrics = extract_local_features(code)
            response['local_metrics'] = local_metrics
        
        response.update({
            'analysis_id': analysis_id,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Sauvegarder
        save_analysis_to_history(code, response, session.get('user_id'))
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"API analyze error {analysis_id}: {e}")
        return jsonify({
            'error': str(e),
            'analysis_id': analysis_id,
            'type': 'internal_error'
        }), 500


@app.route('/debug')
def debug():
    """Page de debug pour tester les métriques localement"""
    local_available = HAS_RADON
    return render_template('debug.html', local_available=local_available)


@app.route('/api/debug', methods=['POST'])
def api_debug():
    """Debug local des métriques avec validation"""
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code:
            return jsonify({'error': 'Code requis'}), 400
        
        if not HAS_RADON:
            return jsonify({'error': 'Radon non disponible pour l\'analyse locale'}), 500
        
        features, metrics = extract_local_features(code)
        
        if features is None:
            return jsonify({'error': 'Extraction des features échouée'}), 500
        
        # Calcul des seuils
        thresholds = {
            'n_threshold': features[2] >= 300,
            'v_threshold': features[3] >= 1000,
            'd_threshold': features[5] >= 50,
            'effort_threshold': features[7] >= 500000,
            'time_threshold': features[8] >= 5000,
            'evaluation': int(features[17])
        }
        
        # Recommandations basées sur les métriques
        recommendations = []
        if thresholds['n_threshold']:
            recommendations.append("Code très long - Considérer la refactorisation")
        if thresholds['v_threshold']:
            recommendations.append("Volume élevé - Simplifier la logique")
        if thresholds['d_threshold']:
            recommendations.append("Code difficile à comprendre - Améliorer la lisibilité")
        if metrics['comments_ratio'] < 10:
            recommendations.append("Ajouter plus de commentaires")
        
        return jsonify({
            'features': features,
            'metrics': metrics,
            'thresholds': thresholds,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Status complet de l'application et de l'API"""
    api_health = client.health()
    status_info = client.status() if api_health else {}
    models_info = client.get_models_info() if api_health else {}
    
    app_status = {
        'web_app': {
            'status': 'running',
            'version': '2.0.0',
            'radon_available': HAS_RADON,
            'upload_folder': str(UPLOAD_FOLDER),
            'max_file_size': MAX_FILE_SIZE,
            'cache_duration_hours': CACHE_DURATION.total_seconds() / 3600
        },
        'api': {
            'health': api_health,
            'info': status_info,
            'models': models_info
        },
        'system': {
            'timestamp': datetime.now().isoformat(),
            'session_id': session.get('user_id', 'no-session'),
            'cache_entries': len(client.cache)
        }
    }
    
    return jsonify(app_status)


@app.route('/history')
def history():
    """Page d'historique des analyses"""
    user_history = get_analysis_history(20)
    return render_template('history.html', analyses=user_history)


@app.route('/api/history')
def api_history():
    """API pour récupérer l'historique"""
    limit = request.args.get('limit', 10, type=int)
    limit = min(limit, 50)  # Limiter à 50 max
    
    history = get_analysis_history(limit)
    return jsonify({
        'analyses': history,
        'total': len(history),
        'limit': limit
    })


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Vider le cache de l'application"""
    try:
        client.cache.clear()
        cache_file = PROJECT_ROOT / 'temp_cache' / 'analysis_history.json'
        if cache_file.exists():
            cache_file.unlink()
        
        return jsonify({'message': 'Cache vidé avec succès'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Page d'upload de fichiers Python"""
    if request.method == 'POST':
        return analyze()  # Rediriger vers la fonction d'analyse
    
    return render_template('upload.html')


@app.route('/about')
def about():
    """Page à propos avec informations techniques"""
    return render_template('about.html')


@app.route('/api/metrics/<analysis_id>')
def get_analysis_metrics(analysis_id):
    """Récupérer les métriques détaillées d'une analyse spécifique"""
    try:
        history = get_analysis_history(100)
        analysis = next((a for a in history if a.get('id') == analysis_id), None)
        
        if not analysis:
            return jsonify({'error': 'Analyse non trouvée'}), 404
        
        return jsonify(analysis), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Gestionnaires d'erreurs avancés
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"Page non trouvée: {request.path}")
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page non trouvée"), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Erreur serveur: {error}")
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Erreur serveur"), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'Fichier trop volumineux'}), 413


# Middleware pour la sécurité
@app.after_request
def after_request(response):
    # Headers de sécurité
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # CORS pour l'API
    if request.path.startswith('/api/'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response


if __name__ == '__main__':
    logger.info("🐛 Bug Predictor Web App - Starting Advanced Version")
    logger.info("🔧 Configuration:")
    logger.info(f"   - API URL: {API_URL}")
    logger.info(f"   - Upload Folder: {UPLOAD_FOLDER}")
    logger.info(f"   - Max File Size: {MAX_FILE_SIZE} bytes")
    logger.info(f"   - Cache Duration: {CACHE_DURATION}")
    logger.info(f"   - Radon Available: {HAS_RADON}")
    logger.info("🌐 Available at: http://localhost:8081")
    
    app.run(host='0.0.0.0', port=8081, debug=True, threaded=True)