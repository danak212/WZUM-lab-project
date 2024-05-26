import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nba_api.stats.endpoints import leaguedashplayerstats
import joblib
from sklearn.preprocessing import StandardScaler

def fetch_data(season='2023-24'):
    print("Fetching data...")
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    print("Data fetched successfully")
    return data

def preprocess_data(data):
    print("Preprocessing data...")
    data = data.fillna(0)  # Wypełnianie brakujących wartości zerami
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    labels = data['PLAYER_ID']  # Zmienna docelowa
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Data preprocessed successfully")
    return features, labels

def train_model(features, labels):
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model

def save_model(model, file_name):
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")

def main(model_file):
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