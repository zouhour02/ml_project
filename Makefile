# Définition des fichiers et chemins
PREPARED_DATA_FILE = /home/zouhour/ml_project/prepared_data.pkl
MODEL_FILE = /home/zouhour/ml_project/model.pkl

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

# Charger le modèle
load:
	python3 main.py --load

# Exécuter les tests unitaires
test:
	pytest tests/

# Nettoyage des fichiers temporaires
clean:
	rm -rf __pycache__ *.pkl venv

