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

## Api

```bash
uvicorn api.app.main:app --reload
```


