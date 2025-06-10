import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def load_and_preprocess_data(path='data/zoo.csv'):
    df = pd.read_csv(path)
    X = df.drop(['animal_name', 'class_type'], axis=1)
    y = df['class_type'] - 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_encoded = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test), scaler