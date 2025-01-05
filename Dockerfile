# Image de base compatible
FROM python:3.10-slim

# Répertoire de travail
WORKDIR /app

# Copier le projet
COPY . .

# Installer les dépendances
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Commande par défaut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

