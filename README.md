# WZUM-lab-project

____________________________________________________________________________________________________
 Machine learning project predicting the first three teams of the NBA season

____________________________________________________________________________________________________
The goal of this project is to predict the players who will receive awards at the end of the NBA regular season, specifically those who will be selected for the All-NBA Team (three teams) and the All-Rookie Team (two teams). The project utilizes detailed player statistics collected throughout the 82-game regular season.

____________________________________________________________________________________________________
Key points:
1. Uses the nba_api database from the repository: nba_api GitHub: https://github.com/swar/nba_api
2. Trains a model to make predictions based on player performance data.

____________________________________________________________________________________________________
How to Run the Project

1. Clone the repository:

_git clone https://github.com/danak212/WZUM-lab-project/_
_cd <repository_directory>_

3. Set up the virtual environment:

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

4. Run the training script:

python train_model.py model.pkl

6. Run the prediction script:

python predict_model.py model.pkl results.json

8. Ensure output is properly configured:

Make sure to add the parameter results.json in the project configuration to display the output data correctly.

____________________________________________________________________________________________________
This will fetch the necessary data, train the model, and predict the players for the All-NBA and All-Rookie teams based on the 2023-24 season statistics.
