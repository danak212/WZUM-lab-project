import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nba_api.stats.endpoints import leaguedashplayerstats
import json
from sklearn.preprocessing import StandardScaler


def fetch_data(season='2023-24'):
    # Pobieranie danych z API NBA
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    return data


def preprocess_data(data):
    data = data.fillna(0)  # Wypełnianie brakujących wartości zerami
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    labels = data['PLAYER_ID']  # Zmienna docelowa - należy dostosować do faktycznych etykiet
    # Normalizacja cech
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels


def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict(model, features):
    predictions = model.predict(features)
    return predictions

def rank_players(data):
    # Ranking graczy na podstawie statystyk
    data['RANK'] = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']].sum(axis=1)
    ranked_players = data.sort_values(by='RANK', ascending=False)
    return ranked_players


def select_teams(ranked_players):
    first_team = ranked_players.head(5)['PLAYER_NAME'].tolist()
    second_team = ranked_players.iloc[5:10]['PLAYER_NAME'].tolist()
    third_team = ranked_players.iloc[10:15]['PLAYER_NAME'].tolist()
    first_rookie_team = ranked_players.iloc[15:20]['PLAYER_NAME'].tolist()
    second_rookie_team = ranked_players.iloc[20:25]['PLAYER_NAME'].tolist()

    results = {
        "first all-nba team": first_team,
        "second all-nba team": second_team,
        "third all-nba team": third_team,
        "first rookie all-nba team": first_rookie_team,
        "second rookie all-nba team": second_rookie_team
    }
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f)

def main(output_file):
    data = fetch_data()
    features, labels = preprocess_data(data)
    model = train_model(features, labels)
    predictions = predict(model, features)
    ranked_players = rank_players(data)
    results = select_teams(ranked_players)
    save_results(results, output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <output_file>")
        sys.exit(1)
    output_file = sys.argv[1]
    main(output_file)
