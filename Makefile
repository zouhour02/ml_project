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


