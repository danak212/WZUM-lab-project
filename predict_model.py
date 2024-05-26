import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.preprocessing import StandardScaler
import joblib
import json

def fetch_data(season='2023-24'):
    # Fetch player statistics from the NBA API
    # Pobieranie statystyk zawodników z interfejsu API NBA
    print("Fetching NBA API data...")
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    print("Data fetched successfully from NBA API")
    return data

def preprocess_data(data):
    # Preprocess the data by filling missing values and standardizing the features
    # Przetwarzanie danych poprzez wypełnianie brakujących wartości i standaryzację cech
    print("Preprocessing data...")
    data = data.fillna(0)
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Data preprocessed successfully")
    return features, data

def rank_players(data):
    # Rank the players based on their combined statistics
    # Ranking zawodników na podstawie ich skumulowanych statystyk
    print("Ranking players...")
    data['RANK'] = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']].sum(axis=1)
    ranked_players = data.sort_values(by='RANK', ascending=False)
    print("Players ranked successfully")
    return ranked_players

def select_teams(ranked_players):
    # Select the top players for each of the teams
    # Wybór najlepszych zawodników do każdej z drużyn
    print("Selecting teams...")

    first_team = ranked_players.head(5)['PLAYER_NAME'].tolist()
    second_team = ranked_players.iloc[5:10]['PLAYER_NAME'].tolist()
    third_team = ranked_players.iloc[10:15]['PLAYER_NAME'].tolist()

    # Select rookie teams based on the remaining top players
    # Wybór drużyn debiutantów na podstawie pozostałych najlepszych zawodników
    rookie_players = ranked_players.iloc[15:25]['PLAYER_NAME'].tolist()
    first_rookie_team = rookie_players[:5]
    second_rookie_team = rookie_players[5:]

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
    # Save the results to a JSON file
    # Zapis wyników do pliku JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved successfully in {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main(model_file, output_file):
    # Main function to load the model, fetch data, preprocess, rank players, select teams, and save results
    # Główna funkcja do ładowania modelu, pobierania danych, przetwarzania, rankowania zawodników, wybierania drużyn i zapisywania wyników
    model = joblib.load(model_file)
    data = fetch_data()
    features, processed_data = preprocess_data(data)
    ranked_players = rank_players(processed_data)
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
