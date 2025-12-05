/**
 * Bug Predictor AI - Main JavaScript
 * Gère les fonctionnalités générales de l'interface VS Code
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialisation de l'interface
    initializeInterface();
    initializeTheme();
    initializeShortcuts();
    initializeActivityBar();
});

function initializeInterface() {
    // Mise à jour de l'horloge dans la barre de statut
    updateClock();
    setInterval(updateClock, 60000);
    
    // Initialisation des tooltips Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialisation des popovers Bootstrap
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    const clockElement = document.getElementById('current-time');
    if (clockElement) {
        clockElement.textContent = timeString;
    }
}

function initializeTheme() {
    // Gestion du thème sombre/clair (VS Code style)
    const savedTheme = localStorage.getItem('vscode-theme') || 'dark';
    applyTheme(savedTheme);
}

function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('vscode-theme', theme);
}

function initializeShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Shift + P : Command Palette
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'P') {
            e.preventDefault();
            toggleCommandPalette();
        }
        
        // Ctrl/Cmd + S : Sauvegarder (simulé)
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            simulateSave();
        }
        
        // Ctrl/Cmd + Enter : Analyser le code
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            triggerAnalysis();
        }
        
        // Escape : Fermer la palette de commandes
        if (e.key === 'Escape') {
            closeCommandPalette();
        }
        
        // F5 : Refresh/Reload
        if (e.key === 'F5') {
            e.preventDefault();
            refreshAnalysis();
        }
    });
}

function toggleCommandPalette() {
    const palette = document.getElementById('command-palette');
    if (palette) {
        palette.classList.toggle('d-none');
        if (!palette.classList.contains('d-none')) {
            const input = palette.querySelector('.command-input');
            if (input) {
                input.focus();
                input.value = '';
            }
            loadCommands();
        }
    }
}

function closeCommandPalette() {
    const palette = document.getElementById('command-palette');
    if (palette) {
        palette.classList.add('d-none');
    }
}

function loadCommands() {
    const commands = [
        { name: 'Analyser le code', action: 'analyze', icon: 'fa-play' },
        { name: 'Charger exemple propre', action: 'example-clean', icon: 'fa-check' },
        { name: 'Charger exemple complexe', action: 'example-complex', icon: 'fa-exclamation-triangle' },
        { name: 'Effacer le code', action: 'clear', icon: 'fa-trash' },
        { name: 'Aller au debug', action: 'debug', icon: 'fa-bug' },
        { name: 'Documentation', action: 'about', icon: 'fa-book' },
        { name: 'Vérifier statut API', action: 'api-status', icon: 'fa-server' }
    ];
    
    const resultsContainer = document.querySelector('.command-results');
    if (resultsContainer) {
        resultsContainer.innerHTML = commands.map(cmd => `
            <div class="command-item" data-action="${cmd.action}">
                <i class="fas ${cmd.icon} me-2"></i>
                ${cmd.name}
            </div>
        `).join('');
        
        // Ajouter les événements de clic
        resultsContainer.querySelectorAll('.command-item').forEach(item => {
            item.addEventListener('click', () => executeCommand(item.dataset.action));
        });
    }
}

function executeCommand(action) {
    closeCommandPalette();
    
    switch (action) {
        case 'analyze':
            triggerAnalysis();
            break;
        case 'example-clean':
            if (window.codeAnalyzer) window.codeAnalyzer.loadExample('clean');
            break;
        case 'example-complex':
            if (window.codeAnalyzer) window.codeAnalyzer.loadExample('complex');
            break;
        case 'clear':
            if (window.codeAnalyzer) window.codeAnalyzer.clearCode();
            break;
        case 'debug':
            window.location.href = '/debug';
            break;
        case 'about':
            window.location.href = '/about';
            break;
        case 'api-status':
            if (window.codeAnalyzer) window.codeAnalyzer.checkApiStatus();
            break;
    }
}

function simulateSave() {
    showNotification('Code sauvegardé (simulation)', 'success');
}

function triggerAnalysis() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn && !analyzeBtn.disabled) {
        analyzeBtn.click();
    }
}

function refreshAnalysis() {
    if (window.codeAnalyzer) {
        window.codeAnalyzer.checkApiStatus();
        showNotification('Statut API actualisé', 'info');
    }
}

function initializeActivityBar() {
    const activityItems = document.querySelectorAll('.activity-item');
    activityItems.forEach(item => {
        item.addEventListener('click', function() {
            // Retirer la classe active de tous les items
            activityItems.forEach(i => i.classList.remove('active'));
            // Ajouter la classe active à l'item cliqué
            this.classList.add('active');
        });
    });
    
    // Navigation dans la sidebar
    const sidebarItems = document.querySelectorAll('.sidebar-item[href]');
    sidebarItems.forEach(item => {
        item.addEventListener('click', function(e) {
            // Retirer la classe active de tous les items
            sidebarItems.forEach(i => i.classList.remove('active'));
            // Ajouter la classe active à l'item cliqué
            this.classList.add('active');
        });
    });
}

function showNotification(message, type = 'info') {
    // Créer une notification temporaire
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-suppression après 3 secondes
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 3000);
}

// Gestion des erreurs globales
window.addEventListener('error', function(e) {
    console.error('Erreur JavaScript:', e.error);
    showNotification('Une erreur JavaScript est survenue', 'danger');
});

// Gestion des erreurs de réseau
window.addEventListener('unhandledrejection', function(e) {
    console.error('Promesse rejetée:', e.reason);
    if (e.reason && e.reason.message && e.reason.message.includes('fetch')) {
        showNotification('Erreur de connexion réseau', 'warning');
    }
});

// Fonctions utilitaires
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Export des fonctions pour utilisation globale
window.VSCodeInterface = {
    showNotification,
    executeCommand,
    toggleCommandPalette,
    applyTheme,
    debounce,
    throttle
};