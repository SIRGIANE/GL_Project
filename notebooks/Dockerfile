FROM jupyter/scipy-notebook:latest

USER root

# Install additional system packages
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

# Install additional Python packages for ML and code analysis
RUN pip install --no-cache-dir \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    GitPython==3.1.30 \
    pydriller==2.5 \
    lizard==1.17.10 \
    radon==6.0.1 \
    nltk==3.8.1 \
    textblob==0.17.1 \
    psycopg2-binary==2.9.7

# Set working directory
WORKDIR /home/jovyan/work