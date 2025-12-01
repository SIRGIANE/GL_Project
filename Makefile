.PHONY: help build up down logs shell clean dev prod

help: ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Construire tous les conteneurs
	docker-compose build

dev: ## Lancer en mode développement
	docker-compose -f docker-compose.dev.yml up --build

up: ## Lancer tous les services
	docker-compose up -d

down: ## Arrêter tous les services
	docker-compose down

logs: ## Voir les logs de tous les services
	docker-compose logs -f

logs-backend: ## Voir les logs du backend
	docker-compose logs -f backend

logs-frontend: ## Voir les logs du frontend
	docker-compose logs -f frontend

shell-backend: ## Accéder au shell du backend
	docker-compose exec backend bash

shell-db: ## Accéder à PostgreSQL
	docker-compose exec db psql -U postgres -d risk_predictor

jupyter: ## Ouvrir Jupyter Lab
	@echo "Jupyter Lab sera accessible sur: http://localhost:8888"
	docker-compose up -d jupyter

restart: ## Redémarrer tous les services
	docker-compose restart

clean: ## Nettoyer les conteneurs et volumes
	docker-compose down -v
	docker system prune -f

reset-db: ## Réinitialiser la base de données
	docker-compose down
	docker volume rm gl_project_postgres_data || true
	docker-compose up -d db

install-frontend: ## Installer les dépendances frontend
	cd frontend && npm install

create-env: ## Créer le fichier .env à partir de .env.example
	cp .env.example .env

setup: create-env build ## Configuration initiale complète
	@echo "Configuration terminée! Utilisez 'make dev' pour démarrer en mode développement."