#!/usr/bin/env python3
"""
Script de démarrage pour Bug Predictor AI
Lance l'API et l'application web Flask
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def check_requirements():
    """Vérifie que les dépendances sont installées"""
    try:
        import flask
        import radon
        print("✓ Dépendances vérifiées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez les dépendances avec: pip install -r requirements.txt")
        return False

def start_api():
    """Lance l'API en arrière-plan"""
    print("🚀 Démarrage de l'API...")
    api_process = subprocess.Popen([
        sys.executable, "app/api.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Attendre que l'API soit prête
    time.sleep(3)
    
    if api_process.poll() is None:
        print("✓ API démarrée sur http://localhost:5001")
        return api_process
    else:
        print("❌ Échec du démarrage de l'API")
        return None

def start_web_app():
    """Lance l'application web"""
    print("🌐 Démarrage de l'application web...")
    web_process = subprocess.Popen([
        sys.executable, "app/web_app.py"
    ])
    
    time.sleep(2)
    
    if web_process.poll() is None:
        print("✓ Application web démarrée sur http://localhost:8081")
        return web_process
    else:
        print("❌ Échec du démarrage de l'application web")
        return None

def main():
    print("=" * 60)
    print("🐛 BUG PREDICTOR AI - DÉMARRAGE")
    print("=" * 60)
    
    # Vérifier le répertoire de travail
    if not Path("app/web_app.py").exists():
        print("❌ Erreur: Lancez ce script depuis le répertoire racine du projet")
        sys.exit(1)
    
    # Vérifier les dépendances
    if not check_requirements():
        sys.exit(1)
    
    processes = []
    
    try:
        # Démarrer l'API
        api_process = start_api()
        if api_process:
            processes.append(api_process)
        
        # Démarrer l'application web
        web_process = start_web_app()
        if web_process:
            processes.append(web_process)
        
        if not processes:
            print("❌ Aucun service n'a pu être démarré")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("✅ SERVICES DÉMARRÉS AVEC SUCCÈS!")
        print("=" * 60)
        print("📡 API Backend:      http://localhost:5001")
        print("🌐 Application Web:  http://localhost:8081")
        print("=" * 60)
        print("\nAppuyez sur Ctrl+C pour arrêter tous les services...")
        
        # Attendre l'interruption
        while True:
            time.sleep(1)
            
            # Vérifier si les processus sont encore en vie
            for i, process in enumerate(processes[:]):
                if process.poll() is not None:
                    print(f"⚠️  Un service s'est arrêté (code: {process.returncode})")
                    processes.remove(process)
            
            if not processes:
                print("❌ Tous les services se sont arrêtés")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt des services...")
        
        # Arrêter tous les processus
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print("✓ Service arrêté")
            except subprocess.TimeoutExpired:
                process.kill()
                print("✓ Service forcé à s'arrêter")
            except Exception as e:
                print(f"⚠️  Erreur lors de l'arrêt: {e}")
    
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        
        # Nettoyer les processus en cas d'erreur
        for process in processes:
            try:
                process.terminate()
            except:
                pass
        
        sys.exit(1)
    
    print("👋 Au revoir!")

if __name__ == "__main__":
    main()