import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nba_api.stats.endpoints import leaguedashplayerstats
import joblib
from sklearn.preprocessing import StandardScaler

def fetch_data(season='2023-24'):
    # Fetch player statistics from the NBA API
    # Pobieranie statystyk zawodników z interfejsu API NBA
    print("Fetching data...")
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    print("Data fetched successfully")
    return data

def preprocess_data(data):
    # Preprocess the data by filling missing values and standardizing the features
    # Przetwarzanie danych poprzez wypełnianie brakujących wartości i standaryzację cech
    print("Preprocessing data...")
    data = data.fillna(0)
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    labels = data['PLAYER_ID']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Data preprocessed successfully")
    return features, labels

def train_model(features, labels):
    # Train a Random Forest model on the preprocessed data
    # Trenowanie modelu Random Forest na przetworzonych danych
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model

def save_model(model, file_name):
    # Save the trained model to a file
    # Zapis wytrenowanego modelu do pliku
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")

def main(model_file):
    # Main function to fetch data, preprocess, train the model, and save it
    # Główna funkcja do pobierania danych, przetwarzania, trenowania modelu i zapisywania go
    data = fetch_data()
    features, labels = preprocess_data(data)
    model = train_model(features, labels)
    save_model(model, model_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <model_file>")
        sys.exit(1)
    model_file = sys.argv[1]
    main(model_file)