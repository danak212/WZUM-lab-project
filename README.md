#### ======= ENGLISH BELOW =======

# NBA Awards Prediction Project - 2023/2024

## Przegląd Projektu

Celem tego projektu jest przewidzenie składów drużyn All-NBA (trzy piątki) oraz All-Rookie (dwie piątki) na sezon 2023/2024. Predykcje są oparte na statystykach zawodników pobranych z API NBA. Projekt obejmuje trenowanie modelu uczenia maszynowego oraz wykorzystanie tego modelu do przewidywania i rankingu zawodników.

## Struktura Projektu

Projekt składa się z następujących skryptów i plików:
- `main.py`: Orkiestruje procesy trenowania i predykcji.
- `train_model.py`: Trenuje model Random Forest na statystykach zawodników.
- `predict_model.py`: Używa wytrenowanego modelu do przewidywania i rankingu zawodników dla drużyn All-NBA i All-Rookie.
- `model.pkl`: Plik z wytrenowanym modelem.
- `Blaszkiewicz_Daniel.json`: Plik wyjściowy zawierający przewidywane drużyny.

## Zależności

- Python 3.11
- scikit-learn
- nba_api
- joblib
- JSON
- StandardScaler z biblioteki scikit-learn

## Kroki i Implementacja

#### 1. Pobieranie Danych

Dane są pobierane za pomocą biblioteki `nba_api`. Endpoint `leaguedashplayerstats.LeagueDashPlayerStats` jest używany do pobrania statystyk zawodników na sezon 2023/2024.

**Kod:**
```python
from nba_api.stats.endpoints import leaguedashplayerstats

def fetch_data(season='2023-24'):
    print("Pobieranie danych...")
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    print("Dane pobrane pomyślnie")
    return data
```

#### 2. Przetwarzanie Danych

Pobrane dane są przetwarzane przez wypełnienie brakujących wartości zerami oraz standaryzację cech za pomocą `StandardScaler`. Użyte cechy to punkty (PTS), zbiórki (REB), asysty (AST), przechwyty (STL), bloki (BLK), minuty gry (MIN), liczba rozegranych meczów (GP), procent celnych rzutów z gry (FG_PCT), procent celnych rzutów wolnych (FT_PCT) oraz procent celnych rzutów za trzy punkty (FG3_PCT).

**Kod:**
```python
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    print("Przetwarzanie danych...")
    data = data.fillna(0)
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    labels = data['PLAYER_ID']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Dane przetworzone pomyślnie")
    return features, labels
```

#### 3. Trenowanie Modelu

Model Random Forest jest trenowany na przetworzonych danych. Dane są podzielone na zestawy treningowe i testowe w proporcji 80-20. Wytrenowany model jest zapisywany do pliku za pomocą `joblib`.

**Kod:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(features, labels):
    print("Trenowanie modelu...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model wytrenowany pomyślnie")
    return model

def save_model(model, file_name):
    joblib.dump(model, file_name)
    print(f"Model zapisany do {file_name}")
```

#### 4. Predykcja Modelu

Wytrenowany model jest używany do przewidywania i rankingu zawodników na podstawie ich skumulowanych statystyk. Najlepsi zawodnicy są wybierani do drużyn All-NBA i All-Rookie.

**Kod:**
```python
import joblib
import json

def rank_players(data):
    print("Rankowanie zawodników...")
    data['RANK'] = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']].sum(axis=1)
    ranked_players = data.sort_values(by='RANK', ascending=False)
    print("Zawodnicy zrankowani pomyślnie")
    return ranked_players

def select_teams(ranked_players):
    print("Wybór drużyn...")
    first_team = ranked_players.head(5)['PLAYER_NAME'].tolist()
    second_team = ranked_players.iloc[5:10]['PLAYER_NAME'].tolist()
    third_team = ranked_players.iloc[10:15]['PLAYER_NAME'].tolist()
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
    print("Drużyny wybrane pomyślnie")
    return results

def save_results(results, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Wyniki zapisane pomyślnie w {output_file}")
    except Exception as e:
        print(f"Błąd zapisywania wyników: {e}")
```

### Wykonanie

Skrypt `main.py` orkiestruje procesy trenowania i predykcji. Najpierw trenuje model poprzez uruchomienie `train_model.py`, a następnie przewiduje wyniki poprzez uruchomienie `predict_model.py`.

**Kod:**
```python
import os
import subprocess

def main():
    model_file = "model.pkl"
    output_file = "Blaszkiewicz_Daniel.json"
    venv_python = os.path.join("venv", "Scripts", "python.exe")

    print("Trenowanie modelu...")
    train_process = subprocess.run([venv_python, "train_model.py", model_file])
    if train_process.returncode != 0:
        print("Wystąpił błąd podczas trenowania modelu")
        return

    print("Przewidywanie wyników...")
    predict_process = subprocess.run([venv_python, "predict_model.py", model_file, output_file])
    if predict_process.returncode != 0:
        print("Wystąpił błąd podczas przewidywania wyników")
        return

    print("Oba skrypty uruchomione pomyślnie!")

if __name__ == "__main__":
    main()
```

### Jak Uruchomić

1. Upewnij się, że masz zainstalowany Python 3.11.
2. Zainstaluj wymagane biblioteki używając `pip install -r requirements.txt`.
3. Ustaw wirtualne środowisko i aktywuj je.
4. Uruchom skrypt `main.py`:
    ```
    python main.py
    ```

### Wynik

Wynik zostanie zapisany w pliku JSON o nazwie `Blaszkiewicz_Daniel.json`, zawierającym przewidywane drużyny All-NBA i All-Rookie w określonym formacie.

### Podsumowanie

Ten projekt demonstruje wykorzystanie uczenia maszynowego do przewidywania nagród NBA na podstawie statystyk zawodników. Proces obejmuje pobieranie danych, przetwarzanie, trenowanie modelu oraz przewidywanie, co skutkuje listą przewidywanych zwycięzców nagród.

#
#

#### ======= ENGLISH VERSION =======

# NBA Awards Prediction Project - 2023/2024

## Project Overview

The goal of this project is to predict the All-NBA (three teams) and All-Rookie (two teams) teams for the 2023/2024 season. The predictions are based on player statistics fetched from the NBA API. The project involves training a machine learning model and using this model to predict and rank players.

## Project Structure

The project consists of the following scripts and files:
- `main.py`: Orchestrates the training and prediction processes.
- `train_model.py`: Trains a Random Forest model on player statistics.
- `predict_model.py`: Uses the trained model to predict and rank players for the All-NBA and All-Rookie teams.
- `model.pkl`: File with the trained model.
- `Blaszkiewicz_Daniel.json`: Output file containing the predicted teams.

## Dependencies

- Python 3.11
- scikit-learn
- nba_api
- joblib
- JSON
- StandardScaler from scikit-learn

## Steps and Implementation

#### 1. Fetching Data

Data is fetched using the `nba_api` library. The `leaguedashplayerstats.LeagueDashPlayerStats` endpoint is used to fetch player statistics for the 2023/2024 season.

**Code:**
```python
from nba_api.stats.endpoints import leaguedashplayerstats

def fetch_data(season='2023-24'):
    print("Fetching data...")
    data = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    print("Data fetched successfully")
    return data
```

#### 2. Data Processing

Fetched data is processed by filling missing values with zeros and standardizing features using `StandardScaler`. Features used are points (PTS), rebounds (REB), assists (AST), steals (STL), blocks (BLK), minutes played (MIN), games played (GP), field goal percentage (FG_PCT), free throw percentage (FT_PCT), and three-point percentage (FG3_PCT).

**Code:**
```python
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    print("Processing data...")
    data = data.fillna(0)
    features = data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    labels = data['PLAYER_ID']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Data processed successfully")
    return features, labels
```

#### 3. Training the Model

A Random Forest model is trained on the processed data. The data is split into training and test sets in an 80-20 ratio. The trained model is saved to a file using `joblib`.

**Code:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(features, labels):
    print("Training the model...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model

def save_model(model, file_name):
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")
```

#### 4. Model Prediction

The trained model is used to predict and rank players based on their cumulative statistics. The top players are selected for the All-NBA and All-Rookie teams.

**Code:**
```python
import joblib
import json

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
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
```

### Execution

The `main.py` script orchestrates the training and prediction processes. It first trains the model by running `train_model.py`, then predicts the results by running `predict_model.py`.

**Code:**
```python
import os
import subprocess

def main():
    model_file = "model.pkl"
    output_file = "Blaszkiewicz_Daniel.json"
    venv_python = os.path.join("venv", "Scripts", "python.exe")

    print("Training the model...")
    train_process = subprocess.run([venv_python, "train_model.py", model_file])
    if train_process.returncode != 0:
        print("Error occurred while training the model")
        return

    print("Predicting results...")
    predict_process = subprocess.run([venv_python, "predict_model.py", model_file, output_file])
    if predict_process.returncode != 0:
        print("Error occurred while predicting results")
        return

    print("Both scripts ran successfully!")

if __name__ == "__main__":
    main()
```

### How to Run

1. Ensure you have Python 3.11 installed.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Set up and activate a virtual environment.
4. Run the `main.py` script:
    ```
    python main.py
    ```

### Output

The output will be saved in a JSON file named `Blaszkiewicz_Daniel.json`, containing the predicted All-NBA and All-Rookie teams in a specified format.

### Summary

This project demonstrates the use of machine learning to predict NBA awards based on player statistics. The process involves data fetching, processing, model training, and prediction, resulting in a list of predicted award winners.
