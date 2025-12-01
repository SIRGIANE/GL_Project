-- Initialisation de la base de données pour Risk Predictor

-- Table pour stocker les projets analysés
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    repository_url VARCHAR(500),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analysis TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Table pour stocker les fichiers analysés
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    file_path VARCHAR(1000) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_extension VARCHAR(10),
    lines_of_code INTEGER,
    complexity_score FLOAT,
    bug_prediction_score FLOAT,
    risk_level VARCHAR(20), -- low, medium, high
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour stocker les métriques des fichiers
CREATE TABLE IF NOT EXISTS file_metrics (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour stocker l'historique des prédictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    model_version VARCHAR(50),
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour optimiser les requêtes
CREATE INDEX IF NOT EXISTS idx_files_project_id ON files(project_id);
CREATE INDEX IF NOT EXISTS idx_file_metrics_file_id ON file_metrics(file_id);
CREATE INDEX IF NOT EXISTS idx_predictions_project_id ON predictions(project_id);