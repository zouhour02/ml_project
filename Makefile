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


#-----------------------Docker---------------------------
# Variables
IMAGE_NAME=zouhour451/mlflow-flask
CONTAINER_NAME=mlflow_flask_container
PORTS=-p 5000:5000 -p 5001:5001
VOLUME=-v ~/mlruns:/home/zouhour/mlruns

# Pull the latest image from Docker Hub
pull:
	docker pull $(IMAGE_NAME):latest

# Stop & remove the existing container if running
stop:
	-docker stop $(CONTAINER_NAME) || true
	-docker rm $(CONTAINER_NAME) || true

# Run the container
run: stop
	docker run -d $(PORTS) --name $(CONTAINER_NAME) $(VOLUME) $(IMAGE_NAME):latest

# Full deployment: Pull, stop old container, and run
deploy: pull run

# Show logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop & remove the container completely
clean: stop
	docker rmi $(IMAGE_NAME):latest


