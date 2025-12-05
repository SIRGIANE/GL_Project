/**
 * Bug Predictor AI - Debug JavaScript
 * Gère l'analyse locale des métriques de code sans prédiction ML
 */

class DebugAnalyzer {
    constructor() {
        this.webApiUrl = '/api';
        this.chart = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadExamples();
    }

    bindEvents() {
        // Bouton d'analyse debug
        document.getElementById('debug-analyze-btn')?.addEventListener('click', (e) => this.handleDebugAnalyze(e));

        // Boutons d'exemple
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.loadExample(e.target.getAttribute('data-example')));
        });

        // Bouton clear
        document.getElementById('clear-debug-code')?.addEventListener('click', () => this.clearCode());
    }

    loadExamples() {
        this.examples = {
            clean: `def calculate_factorial(n):
    """
    Calcule la factorielle d'un nombre.
    Simple et bien structuré.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n doit être un entier positif")
    
    if n <= 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


def is_prime(n):
    """Vérifie si un nombre est premier"""
    if n < 2:
        return False
    
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    
    return True


def main():
    number = 5
    print(f"Factorielle de {number}: {calculate_factorial(number)}")
    print(f"{number} est premier: {is_prime(number)}")


if __name__ == "__main__":
    main()`,

            complex: `# Code complexe avec de nombreuses variables et structures imbriquées
def complex_processor(data_input, config_params={}, debug_mode=False, validation_level=3):
    # Initialisation de nombreuses variables
    temp_storage = {}
    processing_cache = []
    error_tracking = {"errors": 0, "warnings": 0, "info": 0}
    validation_rules = {"strict": True, "tolerant": False, "experimental": False}
    
    # Boucles imbriquées complexes
    for primary_index in range(0, 100):
        for secondary_index in range(0, 50):
            for tertiary_index in range(0, 25):
                for quaternary_index in range(0, 10):
                    # Conditions multiples imbriquées
                    if primary_index > 50:
                        if secondary_index > 25:
                            if tertiary_index > 12:
                                if quaternary_index > 5:
                                    complex_calculation = (primary_index * secondary_index * tertiary_index * quaternary_index)
                                    if complex_calculation > 1000000:
                                        temp_storage[f"key_{primary_index}_{secondary_index}"] = complex_calculation / 1000
                                    elif complex_calculation > 100000:
                                        temp_storage[f"key_{primary_index}_{secondary_index}"] = complex_calculation / 100
                                    elif complex_calculation > 10000:
                                        temp_storage[f"key_{primary_index}_{secondary_index}"] = complex_calculation / 10
                                    else:
                                        temp_storage[f"key_{primary_index}_{secondary_index}"] = complex_calculation
                                else:
                                    processing_cache.append({"pi": primary_index, "si": secondary_index, "ti": tertiary_index})
                            else:
                                error_tracking["warnings"] += 1
                        else:
                            error_tracking["info"] += 1
                    else:
                        if debug_mode:
                            print(f"Processing index {primary_index}")
    
    # Traitement final avec de nombreuses opérations
    final_result = 0
    for key, value in temp_storage.items():
        if isinstance(value, (int, float)):
            final_result += value * 1.5 + 2.7 - 0.3 / 1.1 * 2.2 + 3.3 - 1.8
    
    return {"result": final_result, "cache": processing_cache, "errors": error_tracking}`
        };
    }

    loadExample(type) {
        const codeEditor = document.getElementById('debug-code');
        if (codeEditor && this.examples[type]) {
            codeEditor.value = this.examples[type];
            this.updateDebugConsole(`Exemple ${type} chargé`, 'SUCCESS');
        }
    }

    clearCode() {
        const codeEditor = document.getElementById('debug-code');
        if (codeEditor) {
            codeEditor.value = '';
            this.hideDebugResults();
            this.updateDebugConsole('Code effacé', 'INFO');
        }
    }

    async handleDebugAnalyze(event) {
        event.preventDefault();
        
        const codeEditor = document.getElementById('debug-code');
        const code = codeEditor.value.trim();
        
        if (!code) {
            this.updateDebugConsole('Erreur: Code vide', 'ERROR');
            return;
        }

        this.showDebugLoading();
        this.updateDebugConsole('Démarrage de l\'analyse locale...', 'DEBUG');

        try {
            const response = await fetch(`${this.webApiUrl}/debug`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayDebugResults(result);
                this.updateDebugConsole('Analyse locale terminée', 'SUCCESS');
            } else {
                throw new Error(result.error || 'Erreur d\'analyse locale');
            }
        } catch (error) {
            console.error('Erreur d\'analyse debug:', error);
            this.updateDebugConsole(`Erreur: ${error.message}`, 'ERROR');
            this.hideDebugLoading();
        }
    }

    showDebugLoading() {
        document.getElementById('debug-loading')?.classList.remove('d-none');
        document.getElementById('debug-results')?.classList.add('d-none');
        
        const analyzeBtn = document.getElementById('debug-analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        }
    }

    hideDebugLoading() {
        document.getElementById('debug-loading')?.classList.add('d-none');
        
        const analyzeBtn = document.getElementById('debug-analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-flask me-2"></i>Run Local Analysis';
        }
    }

    hideDebugResults() {
        document.getElementById('debug-results')?.classList.add('d-none');
    }

    displayDebugResults(result) {
        this.hideDebugLoading();
        
        const debugResultsPanel = document.getElementById('debug-results');
        if (!debugResultsPanel) return;

        // Afficher les seuils
        this.displayThresholds(result.thresholds);

        // Afficher l'évaluation finale
        this.displayFinalEvaluation(result.thresholds);

        // Afficher le tableau des features
        this.displayFeaturesTable(result.features, result.metrics);

        // Créer le graphique
        this.createDebugChart(result.metrics);

        debugResultsPanel.classList.remove('d-none');
        debugResultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    displayThresholds(thresholds) {
        const container = document.getElementById('thresholds-container');
        if (!container) return;

        const thresholdItems = [
            { key: 'n_threshold', name: 'Halstead Length', value: 'n ≥ 300', status: thresholds.n_threshold },
            { key: 'v_threshold', name: 'Volume', value: 'v ≥ 1000', status: thresholds.v_threshold },
            { key: 'd_threshold', name: 'Difficulty', value: 'd ≥ 50', status: thresholds.d_threshold },
            { key: 'effort_threshold', name: 'Effort', value: 'effort ≥ 500k', status: thresholds.effort_threshold },
            { key: 'time_threshold', name: 'Time', value: 'time ≥ 5000', status: thresholds.time_threshold }
        ];

        container.innerHTML = thresholdItems.map(item => {
            const statusClass = item.status ? 'danger' : 'success';
            const statusIcon = item.status ? 'fa-times-circle' : 'fa-check-circle';
            const statusText = item.status ? 'EXCEEDED' : 'OK';

            return `
                <div class="col-md-6 col-lg-4">
                    <div class="card bg-${statusClass} bg-opacity-10 border-${statusClass}">
                        <div class="card-body text-center p-2">
                            <i class="fas ${statusIcon} fa-2x text-${statusClass} mb-2"></i>
                            <h6 class="mb-1">${item.name}</h6>
                            <code class="d-block mb-1">${item.value}</code>
                            <span class="badge bg-${statusClass} small">${statusText}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    displayFinalEvaluation(thresholds) {
        const resultElement = document.getElementById('evaluation-result');
        const explanationElement = document.getElementById('evaluation-explanation');
        
        if (!resultElement || !explanationElement) return;

        const isBuggy = thresholds.evaluation === 1;
        const resultClass = isBuggy ? 'text-danger' : 'text-success';
        const resultIcon = isBuggy ? 'fa-exclamation-triangle' : 'fa-check-circle';
        const resultText = isBuggy ? 'POTENTIELLEMENT BUGGY' : 'CODE SAIN';

        resultElement.innerHTML = `
            <i class="fas ${resultIcon} ${resultClass} me-2"></i>${resultText}
        `;

        const exceededCount = Object.values(thresholds).slice(0, 5).filter(Boolean).length;
        explanationElement.textContent = isBuggy 
            ? `${exceededCount} seuil(s) dépassé(s) → Code complexe détecté`
            : 'Tous les seuils respectés → Code de complexité acceptable';
    }

    displayFeaturesTable(features, metrics) {
        const tableBody = document.querySelector('#debug-features-table tbody');
        if (!tableBody) return;

        const featureDescriptions = [
            { name: 'LOC', desc: 'Lines of Code' },
            { name: 'Cyclomatic Complexity', desc: 'Branching complexity' },
            { name: 'Halstead Length (n)', desc: 'Program length' },
            { name: 'Halstead Volume (v)', desc: 'Program volume' },
            { name: 'Halstead Level (l)', desc: 'Level of abstraction' },
            { name: 'Halstead Difficulty (d)', desc: 'Difficulty to understand' },
            { name: 'Halstead Intelligence (i)', desc: 'Intelligence content' },
            { name: 'Halstead Effort (e)', desc: 'Mental effort required' },
            { name: 'Halstead Time (t)', desc: 'Time to program' },
            { name: 'SLOC', desc: 'Source Lines of Code' },
            { name: 'Comment Lines', desc: 'Number of comment lines' },
            { name: 'Blank Lines', desc: 'Number of blank lines' },
            { name: 'LLOC', desc: 'Logical Lines of Code' },
            { name: 'Unique Operators', desc: 'Number of unique operators' },
            { name: 'Unique Operands', desc: 'Number of unique operands' },
            { name: 'Total Operators', desc: 'Total number of operators' },
            { name: 'Total Operands', desc: 'Total number of operands' },
            { name: 'Evaluation', desc: 'Final complexity evaluation' }
        ];

        tableBody.innerHTML = features.map((value, index) => {
            const feature = featureDescriptions[index];
            const formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
            const statusClass = index === 17 ? (value === 1 ? 'danger' : 'success') : 'secondary';
            const statusText = index === 17 ? (value === 1 ? 'COMPLEX' : 'SIMPLE') : 'OK';

            return `
                <tr>
                    <td><code>f${index + 1}</code></td>
                    <td><strong>${feature.name}</strong></td>
                    <td><code>${formattedValue}</code></td>
                    <td class="small text-vscode-secondary">${feature.desc}</td>
                    <td><span class="badge bg-${statusClass} small">${statusText}</span></td>
                </tr>
            `;
        }).join('');
    }

    createDebugChart(metrics) {
        const canvas = document.getElementById('debugChart');
        if (!canvas) return;

        // Détruire le graphique existant
        if (this.chart) {
            this.chart.destroy();
        }

        const ctx = canvas.getContext('2d');

        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['LOC', 'Complexity', 'Volume', 'Difficulty', 'Effort', 'Time'],
                datasets: [{
                    label: 'Normalized Values',
                    data: [
                        Math.log10(metrics.LOC + 1),
                        Math.log10(metrics.cyclomatic_complexity + 1),
                        Math.log10(metrics.halstead_volume + 1),
                        Math.log10(metrics.halstead_difficulty + 1),
                        Math.log10(metrics.halstead_effort + 1),
                        Math.log10(metrics.halstead_time + 1)
                    ],
                    backgroundColor: [
                        '#007acc', '#89d185', '#ffcc02', '#ff8c00', '#f85149', '#bc89bd'
                    ],
                    borderColor: '#3e3e42',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Code Metrics (Log Scale)',
                        color: '#cccccc'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#969696'
                        },
                        grid: {
                            color: '#3e3e42'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#969696'
                        },
                        grid: {
                            color: '#3e3e42'
                        }
                    }
                }
            }
        });
    }

    updateDebugConsole(message, type = 'INFO') {
        const debugConsole = document.getElementById('debug-console');
        if (debugConsole) {
            const timestamp = new Date().toLocaleTimeString('fr-FR');
            const colorClass = type === 'ERROR' ? 'text-vscode-red' : 
                              type === 'SUCCESS' ? 'text-vscode-green' : 
                              type === 'DEBUG' ? 'text-vscode-yellow' : 'text-vscode-blue';
            
            debugConsole.innerHTML += `<br><span class="${colorClass}">[${type}]</span> <span class="text-vscode-secondary">${timestamp}</span> ${message}`;
            debugConsole.scrollTop = debugConsole.scrollHeight;
        }
    }
}

// Initialiser l'analyseur debug quand le DOM est prêt
document.addEventListener('DOMContentLoaded', () => {
    window.debugAnalyzer = new DebugAnalyzer();
});