# PROJET 7 : Implémentez un modèle de scoring

## Création du projet

```bash
#(Mac)
python3 -m venv venv
source venv/bin/activate
#(Windows)
python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=projet7 --display-name "Python (Projet 7)"
```

## Versionning

```bash
git init
git status
git add .
git commit -m"Initialisation projet"
git remote add origin https://github.com/webew/api-oc-projet7.git
git push -u origin main
```

## MLFlow

Dans un terminal, exécuter :

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Puis accéder à ['MLFLOW'](localhost:5000).

## Api

```bash
uvicorn api.app.main:app --reload
```

## Tests unitaires

```bash
pytest -v
```

## Data drift

Exécuter le fichier artifacts/data_drift_report.html.
