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
	

run-api:
	uvicorn app:app --reload --host 127.0.0.1 --port 8001
	

# Run MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri file:///home/zouhour/mlruns


#-------------------------------------------------------------------------
# Build Images
build-pipeline:
	docker build -t pipeline_image -f Dockerfile.pipeline .

build-fastapi:
	docker build -t fastapi_image -f Dockerfile.fastapi .

build-mlflow:
	docker build -t mlflow_image -f Dockerfile.mlflow .

# Run Containers
run-pipeline:
	docker run --rm pipeline_image

run-fastapi:
	docker run -p 5001:5001 fastapi_image

run-mlflow:
	docker run -p 5000:5000 mlflow_image

# Push to Docker Hub
push-pipeline:
	docker tag pipeline_image zouhour451/pipeline_image
	docker push zouhour451/pipeline_image

push-fastapi:
	docker tag fastapi_image zouhour451/fastapi_image
	docker push zouhour451/fastapi_image

push-mlflow:
	docker tag mlflow_image zouhour451/mlflow_image
	docker push zouhour451/mlflow_image



