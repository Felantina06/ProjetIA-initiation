from utils import download_zoo_dataset
from preprocess import load_and_preprocess_data
from models.model import create_model
import os
import joblib

def main():
    download_zoo_dataset()
    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data()
    model = create_model(input_dim=X_train.shape[1], num_classes=7)
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
    os.makedirs("models", exist_ok=True)
    model.save("models/zoo_model.h5")
    joblib.dump(scaler, "models/scaler.save")
    print("[✓] Modèle et scaler sauvegardés.")

if __name__ == '__main__':
    main()