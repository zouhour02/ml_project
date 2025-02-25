# Définition des fichiers et chemins
PREPARED_DATA_FILE = /home/zouhour/ml_project/prepared_data.pkl
MODEL_FILE = /home/zouhour/ml_project/model.pkl

# Définition des variables

MLFLOW_DB="sqlite:///mlflow.db"

.PHONY:

# Installation des dépendances
install:
	python3 -m venv venv
	bash -i -c "source venv/bin/activate && pip install -r requirements.txt"

# Vérification du code (qualité, formatage, sécurité)
check:
	flake8 .
	black . --check
	bandit -r .

# Préparer les données
prepare:
	python3 main.py --prepare

# Entraîner le modèle
train:
	python3 main.py --train

# Évaluer le modèle
evaluate:
	python3 main.py --evaluate

# Sauvegarder le modèle
save:
	python3 main.py --save
# Full pipeline
all: prepare train evaluate save load

# Charger le modèle
load:
	python3 main.py --load

# Exécuter les tests unitaires
test:
	pytest tests/


# Nettoyage des fichiers temporaires
clean:
	rm -rf __pycache__ *.pkl venv
	
###################### Automatisation CI/CD ###############################

# Check Python code quality, format, and security
check:
	@echo "Running Flake8..."
	flake8 .
	@echo "Running Black..."
	black . --check
	@echo "Running Bandit..."
	bandit -r .

test:
	@echo "Running Tests..."
	pytest tests/

# Watch for Python file changes and run checks and tests
watch:
	@echo "Starting file watcher..."
	while true; do \
		echo "Waiting for Python file changes..."; \
		inotifywait -r -e modify . | grep --line-buffered '\.py$$' && \
		echo "Python file changed, running checks and tests..."; \
		make check test; \
	done


# Watch for data file changes and trigger training, evaluation, and saving
watch-data:
	@echo "Starting data file watcher..."
	while true; do \
		echo "Waiting for data file changes..."; \
		inotifywait -r -e modify . | grep --line-buffered '\\.pkl$' && \
		echo "Data file changed, running model training..." && \
		make train evaluate save; \
	done

# Push Docker image after a commit (if inside a Git repo)
push-auto:
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		make push; \
	else \
		echo "Not inside a Git repository. Skipping push."; \
	fi

run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8001
	

# Run MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri file:///home/zouhour/mlruns


	
	# Construire l'image Docker
build:
	docker build -t zouhour_joudi_4ds1_mlops .

# Exécuter le conteneur Docker
run:
	docker run -p 8000:8000 zouhour_joudi_4ds1_mlops

# Pousser l'image sur Docker Hub
push:
	docker login && docker tag zouhour_joudi_4ds1_mlops zouhour451/zouhour_joudi_4ds1_mlops && docker push zouhour451/zouhour_joudi_4ds1_mlops


