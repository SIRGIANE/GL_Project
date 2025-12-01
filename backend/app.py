from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

@app.route('/')
def hello():
    return jsonify({
        'message': 'Risk Predictor API',
        'status': 'running',
        'version': '1.0.0'
    })

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'database': 'connected',  # TODO: vérifier la connexion DB
        'redis': 'connected'      # TODO: vérifier la connexion Redis
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)