/**
 * Bug Predictor AI - Code Analyzer
 * Gère l'analyse de code en temps réel et l'interaction avec l'API
 */

class CodeAnalyzer {
    constructor() {
        this.apiUrl = 'http://localhost:5001/api';
        this.webApiUrl = '/api';
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiStatus();
        this.loadExamples();
        
        // Vérifier le statut de l'API périodiquement
        setInterval(() => this.checkApiStatus(), 30000);
    }

    bindEvents() {
        // Formulaire d'analyse
        const analyzeForm = document.getElementById('analyze-form');
        if (analyzeForm) {
            analyzeForm.addEventListener('submit', (e) => this.handleAnalyze(e));
        }

        // Boutons d'exemple
        document.getElementById('example-clean')?.addEventListener('click', () => this.loadExample('clean'));
        document.getElementById('example-complex')?.addEventListener('click', () => this.loadExample('complex'));
        document.getElementById('clear-code')?.addEventListener('click', () => this.clearCode());

        // Analyse en temps réel (optionnelle)
        const codeEditor = document.getElementById('code');
        if (codeEditor) {
            let timeout;
            codeEditor.addEventListener('input', () => {
                clearTimeout(timeout);
                timeout = setTimeout(() => this.analyzeRealTime(), 2000);
            });
        }
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.webApiUrl}/status`);
            const data = await response.json();
            
            this.updateApiStatus(data.api_health, data.api_info);
        } catch (error) {
            console.error('Erreur de vérification API:', error);
            this.updateApiStatus(false, {});
        }
    }

    updateApiStatus(isHealthy, info) {
        const statusElement = document.getElementById('api-status-detail');
        const indicator = document.getElementById('api-indicator');
        const text = document.getElementById('api-text');

        if (statusElement) {
            if (isHealthy) {
                statusElement.innerHTML = `
                    <div class="d-flex align-items-center mb-2">
                        <div class="badge badge-success me-2">En ligne</div>
                        <small class="text-vscode-secondary">API opérationnelle</small>
                    </div>
                    <div class="small">
                        <div>Modèles: ${info.models?.total || 0}</div>
                        <div>Version: ${info.api_version || 'N/A'}</div>
                    </div>
                `;
            } else {
                statusElement.innerHTML = `
                    <div class="d-flex align-items-center">
                        <div class="badge badge-danger me-2">Hors ligne</div>
                        <small class="text-vscode-secondary">API indisponible</small>
                    </div>
                `;
            }
        }

        if (indicator) {
            indicator.className = `fas fa-circle me-1 ${isHealthy ? 'text-success' : 'text-danger'}`;
        }

        if (text) {
            text.textContent = isHealthy ? 'API Ready' : 'API Offline';
        }
    }

    loadExamples() {
        this.examples = {
            clean: `def calculate_factorial(n):
    """
    Calcule la factorielle d'un nombre entier positif.
    
    Args:
        n (int): Nombre entier positif
        
    Returns:
        int: Factorielle de n
        
    Raises:
        ValueError: Si n est négatif
        TypeError: Si n n'est pas un entier
    """
    if not isinstance(n, int):
        raise TypeError("n doit être un entier")
    
    if n < 0:
        raise ValueError("n doit être positif")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


def main():
    try:
        number = int(input("Entrez un nombre: "))
        factorial = calculate_factorial(number)
        print(f"La factorielle de {number} est {factorial}")
    except (ValueError, TypeError) as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()`,

            complex: `def process_data(data, config=None, debug=False, verbose=True, options={}):
    results = []
    temp_cache = {}
    global_state = {'processed': 0, 'errors': 0, 'warnings': 0}
    
    if config is None: config = {}
    if not isinstance(data, list): data = [data]
    
    for i, item in enumerate(data):
        try:
            if debug: print(f"Processing item {i}")
            
            if isinstance(item, dict):
                for key in item:
                    if key in temp_cache:
                        value = temp_cache[key]
                    else:
                        if 'transform' in config:
                            if config['transform'] == 'upper':
                                value = str(item[key]).upper()
                            elif config['transform'] == 'lower':
                                value = str(item[key]).lower()
                            elif config['transform'] == 'title':
                                value = str(item[key]).title()
                            else:
                                value = item[key]
                        else:
                            value = item[key]
                        temp_cache[key] = value
                    
                    if 'filter' in options:
                        if options['filter'] == 'numeric':
                            try:
                                float(value)
                            except:
                                continue
                        elif options['filter'] == 'alpha':
                            if not str(value).isalpha():
                                continue
                    
                    processed_value = value
                    if 'multiplier' in config:
                        if isinstance(value, (int, float)):
                            processed_value = value * config['multiplier']
                    
                    if 'prefix' in config:
                        processed_value = config['prefix'] + str(processed_value)
                    
                    if 'suffix' in config:
                        processed_value = str(processed_value) + config['suffix']
                    
                    results.append({
                        'original': item[key],
                        'processed': processed_value,
                        'key': key,
                        'index': i
                    })
                    
                    global_state['processed'] += 1
            else:
                if verbose: print(f"Skipping non-dict item: {item}")
                global_state['warnings'] += 1
                
        except Exception as e:
            if debug: print(f"Error processing item {i}: {e}")
            global_state['errors'] += 1
            continue
    
    if verbose:
        print(f"Processed: {global_state['processed']}, Errors: {global_state['errors']}, Warnings: {global_state['warnings']}")
    
    return results, global_state, temp_cache`
        };
    }

    loadExample(type) {
        const codeEditor = document.getElementById('code');
        if (codeEditor && this.examples[type]) {
            codeEditor.value = this.examples[type];
            this.updateConsole(`[INFO] Exemple de code ${type} chargé`);
            
            // Déclencher l'événement input pour la coloration syntaxique
            codeEditor.dispatchEvent(new Event('input'));
        }
    }

    clearCode() {
        const codeEditor = document.getElementById('code');
        if (codeEditor) {
            codeEditor.value = '';
            this.hideResults();
            this.updateConsole('[INFO] Code effacé');
        }
    }

    async handleAnalyze(event) {
        event.preventDefault();
        
        const codeEditor = document.getElementById('code');
        const code = codeEditor.value.trim();
        
        if (!code) {
            this.showToast('error', 'Veuillez saisir du code à analyser');
            return;
        }

        this.showLoading();
        this.updateConsole('[INFO] Démarrage de l\'analyse...');

        try {
            const response = await fetch(`${this.webApiUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
                this.updateConsole('[SUCCESS] Analyse terminée avec succès');
                this.showToast('success', 'Analyse terminée avec succès');
            } else {
                throw new Error(result.error || 'Erreur d\'analyse');
            }
        } catch (error) {
            console.error('Erreur d\'analyse:', error);
            this.updateConsole(`[ERROR] ${error.message}`);
            this.showToast('error', `Erreur: ${error.message}`);
            this.hideLoading();
        }
    }

    showLoading() {
        document.getElementById('loading-panel')?.classList.remove('d-none');
        document.getElementById('results-panel')?.classList.add('d-none');
        
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        }
    }

    hideLoading() {
        document.getElementById('loading-panel')?.classList.add('d-none');
        
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-play me-2"></i>Run Analysis';
        }
    }

    hideResults() {
        document.getElementById('results-panel')?.classList.add('d-none');
    }

    displayResults(result) {
        this.hideLoading();
        
        const resultsPanel = document.getElementById('results-panel');
        const resultsContent = document.getElementById('results-content');
        
        if (!resultsPanel || !resultsContent) return;

        // Déterminer la classe CSS basée sur le niveau de risque
        const riskClass = this.getRiskClass(result.risk_level);
        const riskIcon = this.getRiskIcon(result.risk_level);

        resultsContent.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card bg-${riskClass} bg-opacity-10 border-${riskClass}">
                        <div class="card-body text-center">
                            <i class="fas ${riskIcon} fa-3x text-${riskClass} mb-3"></i>
                            <h4 class="text-${riskClass}">
                                ${result.is_bug ? 'Bug Détecté' : 'Code Sain'}
                            </h4>
                            <div class="h5">
                                Probabilité: ${(result.probability * 100).toFixed(1)}%
                            </div>
                            <div class="badge bg-${riskClass}">
                                Risque ${result.risk_level}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-chart-bar me-2"></i>Métriques Code
                        </div>
                        <div class="card-body">
                            <div class="row g-2">
                                <div class="col-6">
                                    <div class="text-center p-2 bg-light rounded">
                                        <div class="h6 mb-0">${result.metrics.LOC}</div>
                                        <small>Lignes de Code</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center p-2 bg-light rounded">
                                        <div class="h6 mb-0">${result.metrics.cyclomatic_complexity}</div>
                                        <small>Complexité</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center p-2 bg-light rounded">
                                        <div class="h6 mb-0">${result.metrics.halstead_volume?.toFixed(1) || 'N/A'}</div>
                                        <small>Volume Halstead</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center p-2 bg-light rounded">
                                        <div class="h6 mb-0">${result.metrics.halstead_difficulty?.toFixed(1) || 'N/A'}</div>
                                        <small>Difficulté</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-info-circle me-2"></i>Détails de l'Analyse
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4">
                                    <strong>Modèle utilisé:</strong><br>
                                    <span class="badge bg-secondary">${result.model_used || 'N/A'}</span>
                                </div>
                                <div class="col-md-4">
                                    <strong>Timestamp:</strong><br>
                                    <small class="text-muted">${new Date(result.analysis_timestamp).toLocaleString('fr-FR')}</small>
                                </div>
                                <div class="col-md-4">
                                    <strong>Features extraites:</strong><br>
                                    <small class="text-muted">${result.features?.length || 0} métriques</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        resultsPanel.classList.remove('d-none');
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    getRiskClass(riskLevel) {
        switch (riskLevel?.toUpperCase()) {
            case 'ÉLEVÉ': return 'danger';
            case 'MODÉRÉ': return 'warning';
            case 'FAIBLE': return 'info';
            case 'TRÈS FAIBLE': return 'success';
            default: return 'secondary';
        }
    }

    getRiskIcon(riskLevel) {
        switch (riskLevel?.toUpperCase()) {
            case 'ÉLEVÉ': return 'fa-exclamation-triangle';
            case 'MODÉRÉ': return 'fa-exclamation-circle';
            case 'FAIBLE': return 'fa-info-circle';
            case 'TRÈS FAIBLE': return 'fa-check-circle';
            default: return 'fa-question-circle';
        }
    }

    async analyzeRealTime() {
        // Analyse en temps réel simplifiée (optionnelle)
        const codeEditor = document.getElementById('code');
        const code = codeEditor.value.trim();
        
        if (code.length > 50) { // Seulement pour du code suffisant
            try {
                // Analyse locale basique
                const lines = code.split('\n').length;
                const chars = code.length;
                
                this.updateConsole(`[REALTIME] ${lines} lignes, ${chars} caractères`);
            } catch (error) {
                console.error('Erreur analyse temps réel:', error);
            }
        }
    }

    updateConsole(message) {
        const consoleOutput = document.getElementById('console-output');
        if (consoleOutput) {
            const timestamp = new Date().toLocaleTimeString('fr-FR');
            consoleOutput.innerHTML += `<br><span class="text-vscode-secondary">[${timestamp}]</span> ${message}`;
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
    }

    showToast(type, message) {
        const toastId = type === 'error' ? 'error-toast' : 'success-toast';
        const toast = document.getElementById(toastId);
        
        if (toast) {
            const toastBody = toast.querySelector('.toast-body');
            if (toastBody) {
                toastBody.textContent = message;
            }
            
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
        }
    }
}

// Initialiser l'analyseur quand le DOM est prêt
document.addEventListener('DOMContentLoaded', () => {
    window.codeAnalyzer = new CodeAnalyzer();
});