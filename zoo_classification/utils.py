import os
import pandas as pd

def download_zoo_dataset(path='data/zoo.csv'):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
    columns = [
        "animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
        "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic",
        "catsize", "class_type"
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df = pd.read_csv(url, names=columns)
        df.to_csv(path, index=False)
        print(f"[✓] Dataset téléchargé et sauvegardé à : {path}")
    else:
        print(f"[i] Dataset déjà présent à : {path}")