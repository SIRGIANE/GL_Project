/**
 * Bug Predictor AI - Results JavaScript
 * Gère l'affichage et l'interaction avec les résultats d'analyse
 */

class ResultsManager {
    constructor() {
        this.currentResult = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupCharts();
    }

    bindEvents() {
        // Boutons d'export
        document.getElementById('export-json')?.addEventListener('click', () => this.exportResults('json'));
        document.getElementById('export-csv')?.addEventListener('click', () => this.exportResults('csv'));
        document.getElementById('export-pdf')?.addEventListener('click', () => this.exportResults('pdf'));

        // Bouton de nouvelle analyse
        document.getElementById('new-analysis')?.addEventListener('click', () => this.startNewAnalysis());

        // Boutons de partage
        document.getElementById('share-results')?.addEventListener('click', () => this.shareResults());
        document.getElementById('copy-link')?.addEventListener('click', () => this.copyResultsLink());
    }

    setupCharts() {
        // Configuration des graphiques avec Chart.js
        this.setupMetricsChart();
        this.setupRiskChart();
    }

    setupMetricsChart() {
        const canvas = document.getElementById('metricsChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Données d'exemple - seront remplacées par les vraies données
        this.metricsChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: [
                    'Complexité',
                    'Volume',
                    'Difficulté',
                    'Effort',
                    'Temps',
                    'Maintenabilité'
                ],
                datasets: [{
                    label: 'Métriques Code',
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(0, 122, 204, 0.2)',
                    borderColor: '#007acc',
                    borderWidth: 2,
                    pointBackgroundColor: '#007acc',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#969696',
                            backdropColor: 'transparent'
                        },
                        grid: {
                            color: '#3e3e42'
                        },
                        angleLines: {
                            color: '#3e3e42'
                        },
                        pointLabels: {
                            color: '#cccccc',
                            font: {
                                size: 12
                            }
                        }
                    }
                }
            }
        });
    }

    setupRiskChart() {
        const canvas = document.getElementById('riskChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        this.riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Risque', 'Sain'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#f85149', '#89d185'],
                    borderColor: '#3e3e42',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#cccccc',
                            font: {
                                size: 12
                            }
                        }
                    }
                }
            }
        });
    }

    updateResults(result) {
        this.currentResult = result;
        this.updateMetricsChart(result);
        this.updateRiskChart(result);
        this.updateDetailedMetrics(result);
    }

    updateMetricsChart(result) {
        if (!this.metricsChart || !result.metrics) return;

        // Normaliser les métriques pour le graphique radar (0-100)
        const normalizedData = [
            Math.min(result.metrics.cyclomatic_complexity * 10, 100),
            Math.min(result.metrics.halstead_volume / 10, 100),
            Math.min(result.metrics.halstead_difficulty * 2, 100),
            Math.min(result.metrics.halstead_effort / 10000, 100),
            Math.min(result.metrics.halstead_time / 100, 100),
            Math.max(0, 100 - (result.probability * 100))
        ];

        this.metricsChart.data.datasets[0].data = normalizedData;
        this.metricsChart.update();
    }

    updateRiskChart(result) {
        if (!this.riskChart) return;

        const riskPercentage = result.probability * 100;
        const safePercentage = 100 - riskPercentage;

        this.riskChart.data.datasets[0].data = [riskPercentage, safePercentage];
        this.riskChart.update();
    }

    updateDetailedMetrics(result) {
        // Mettre à jour les métriques détaillées dans l'interface
        const metricsContainer = document.getElementById('detailed-metrics');
        if (!metricsContainer || !result.metrics) return;

        const metrics = result.metrics;
        
        metricsContainer.innerHTML = `
            <div class="row g-3">
                <div class="col-md-4">
                    <div class="metric-card p-3 border rounded">
                        <h6 class="text-vscode-blue mb-2">Lignes de Code</h6>
                        <div class="h5 mb-1">${metrics.LOC}</div>
                        <small class="text-vscode-secondary">Total lines</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card p-3 border rounded">
                        <h6 class="text-vscode-yellow mb-2">Complexité</h6>
                        <div class="h5 mb-1">${metrics.cyclomatic_complexity}</div>
                        <small class="text-vscode-secondary">Cyclomatic complexity</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card p-3 border rounded">
                        <h6 class="text-vscode-green mb-2">Volume Halstead</h6>
                        <div class="h5 mb-1">${metrics.halstead_volume?.toFixed(1) || 'N/A'}</div>
                        <small class="text-vscode-secondary">Program volume</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card p-3 border rounded">
                        <h6 class="text-vscode-orange mb-2">Difficulté</h6>
                        <div class="h5 mb-1">${metrics.halstead_difficulty?.toFixed(1) || 'N/A'}</div>
                        <small class="text-vscode-secondary">Understanding difficulty</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card p-3 border rounded">
                        <h6 class="text-vscode-red mb-2">Effort</h6>
                        <div class="h5 mb-1">${metrics.halstead_effort?.toFixed(0) || 'N/A'}</div>
                        <small class="text-vscode-secondary">Development effort</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card p-3 border rounded">
                        <h6 class="text-vscode-purple mb-2">Temps</h6>
                        <div class="h5 mb-1">${metrics.halstead_time?.toFixed(1) || 'N/A'}s</div>
                        <small class="text-vscode-secondary">Estimated time</small>
                    </div>
                </div>
            </div>
        `;
    }

    exportResults(format) {
        if (!this.currentResult) {
            this.showNotification('Aucun résultat à exporter', 'warning');
            return;
        }

        switch (format) {
            case 'json':
                this.exportAsJSON();
                break;
            case 'csv':
                this.exportAsCSV();
                break;
            case 'pdf':
                this.exportAsPDF();
                break;
        }
    }

    exportAsJSON() {
        const data = {
            timestamp: new Date().toISOString(),
            analysis: this.currentResult,
            metadata: {
                tool: 'Bug Predictor AI',
                version: '1.0.0',
                format: 'json'
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        this.downloadBlob(blob, `bug-analysis-${Date.now()}.json`);
    }

    exportAsCSV() {
        const metrics = this.currentResult.metrics;
        const csvHeader = 'Metric,Value,Description\n';
        
        const csvData = Object.entries(metrics).map(([key, value]) => {
            const description = this.getMetricDescription(key);
            return `"${key}","${value}","${description}"`;
        }).join('\n');

        const csvContent = csvHeader + csvData;
        const blob = new Blob([csvContent], { type: 'text/csv' });
        this.downloadBlob(blob, `bug-analysis-${Date.now()}.csv`);
    }

    exportAsPDF() {
        // Implémentation simplifiée - en production, utiliser une bibliothèque comme jsPDF
        this.showNotification('Export PDF en cours de développement', 'info');
    }

    getMetricDescription(metric) {
        const descriptions = {
            'LOC': 'Total lines of code',
            'cyclomatic_complexity': 'Cyclomatic complexity measure',
            'halstead_volume': 'Halstead volume metric',
            'halstead_difficulty': 'Halstead difficulty measure',
            'halstead_effort': 'Estimated programming effort',
            'halstead_time': 'Estimated programming time'
        };
        return descriptions[metric] || 'Code quality metric';
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showNotification(`Fichier ${filename} téléchargé`, 'success');
    }

    startNewAnalysis() {
        window.location.href = '/';
    }

    shareResults() {
        if (!this.currentResult) return;

        // Créer un lien de partage temporaire
        const shareData = {
            title: 'Bug Predictor AI - Résultats d\'analyse',
            text: `Probabilité de bug: ${(this.currentResult.probability * 100).toFixed(1)}% - Risque ${this.currentResult.risk_level}`,
            url: window.location.href
        };

        if (navigator.share) {
            navigator.share(shareData).catch(console.error);
        } else {
            this.copyToClipboard(shareData.text + '\n' + shareData.url);
            this.showNotification('Résultats copiés dans le presse-papiers', 'success');
        }
    }

    copyResultsLink() {
        const url = window.location.href;
        this.copyToClipboard(url);
        this.showNotification('Lien copié dans le presse-papiers', 'success');
    }

    copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text);
        } else {
            // Fallback pour les navigateurs plus anciens
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
        }
    }

    showNotification(message, type = 'info') {
        // Utiliser le système de notification global
        if (window.VSCodeInterface) {
            window.VSCodeInterface.showNotification(message, type);
        } else {
            alert(message);
        }
    }

    // Méthodes pour l'animation des résultats
    animateResults() {
        const resultElements = document.querySelectorAll('.result-item');
        resultElements.forEach((element, index) => {
            setTimeout(() => {
                element.classList.add('fade-in');
            }, index * 100);
        });
    }

    // Comparaison avec des analyses précédentes
    compareWithPrevious(previousResults) {
        if (!previousResults || !this.currentResult) return;

        const comparison = {
            probability_change: this.currentResult.probability - previousResults.probability,
            complexity_change: this.currentResult.metrics.cyclomatic_complexity - previousResults.metrics.cyclomatic_complexity
        };

        this.displayComparison(comparison);
    }

    displayComparison(comparison) {
        const comparisonContainer = document.getElementById('comparison-results');
        if (!comparisonContainer) return;

        const probabilityTrend = comparison.probability_change > 0 ? 'increase' : 'decrease';
        const complexityTrend = comparison.complexity_change > 0 ? 'increase' : 'decrease';

        comparisonContainer.innerHTML = `
            <div class="alert alert-info">
                <h6><i class="fas fa-chart-line me-2"></i>Comparaison avec l'analyse précédente</h6>
                <p class="mb-1">
                    Probabilité: ${comparison.probability_change > 0 ? '+' : ''}${(comparison.probability_change * 100).toFixed(1)}%
                    <i class="fas fa-arrow-${probabilityTrend === 'increase' ? 'up text-danger' : 'down text-success'} ms-1"></i>
                </p>
                <p class="mb-0">
                    Complexité: ${comparison.complexity_change > 0 ? '+' : ''}${comparison.complexity_change}
                    <i class="fas fa-arrow-${complexityTrend === 'increase' ? 'up text-warning' : 'down text-success'} ms-1"></i>
                </p>
            </div>
        `;
    }
}

// Initialiser le gestionnaire de résultats
document.addEventListener('DOMContentLoaded', () => {
    window.resultsManager = new ResultsManager();
    
    // Si des données de résultats sont disponibles, les charger
    const resultData = document.getElementById('result-data');
    if (resultData) {
        try {
            const result = JSON.parse(resultData.textContent);
            window.resultsManager.updateResults(result);
            window.resultsManager.animateResults();
        } catch (error) {
            console.error('Erreur lors du chargement des résultats:', error);
        }
    }
});