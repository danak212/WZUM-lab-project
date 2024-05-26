import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashplayerstats
import json
import joblib
from sklearn.preprocessing import StandardScaler

def fetch_data(season='2023-24'):
    print("Fetching data...")
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    print("Data fetched successfully")
    return data

def preprocess_data(data):
    print("Preprocessing data...")
    data = data.fillna(0)
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Data preprocessed successfully")
    return features

def rank_players(data):
    print("Ranking players...")
    data['RANK'] = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']].sum(axis=1)
    ranked_players = data.sort_values(by='RANK', ascending=False)
    print("Players ranked successfully")
    return ranked_players

def select_teams(ranked_players):
    print("Selecting teams...")
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
    print("Teams selected successfully")
    return results

def save_results(results, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f)
        print(f"Results saved successfully in {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main(model_file, output_file):
    model = joblib.load(model_file)
    data = fetch_data()
    ranked_players = rank_players(data)
    results = select_teams(ranked_players)
    save_results(results, output_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict_model.py <model_file> <output_file>")
        sys.exit(1)
    model_file = sys.argv[1]
    output_file = sys.argv[2]
    main(model_file, output_file)
