import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

class_labels = {
    1: "Mammifère",
    2: "Oiseau",
    3: "Reptile",
    4: "Poisson",
    5: "Amphibien",
    6: "Insecte",
    7: "Invertébré"
}

def main():
    df = pd.read_csv("test/test_animals.csv")
    animal_names = df['animal_name']
    X = df.drop(['animal_name'], axis=1)
    scaler = joblib.load("models/scaler.save")
    X_scaled = scaler.transform(X)
    model = load_model("models/zoo_model.h5")
    predictions = model.predict(X_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    print("\n📊 Résultats des prédictions :\n")
    for name, pred in zip(animal_names, predicted_classes):
        print(f"Animal: {name} → Classe prédite: {pred + 1} ({class_labels[pred + 1]})")

if __name__ == '__main__':
    main()