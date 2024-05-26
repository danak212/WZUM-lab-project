# Wersja z graczami jak w "wyniki.json"...

import subprocess
import os


def main():
    # Ścieżki do plików modeli i wyników
    model_file = "model.pkl"
    output_file = "blaszkiewicz_daniel.json"

    # Ścieżka do środowiska wirtualnego
    venv_python = os.path.join("venv", "Scripts", "python.exe")

    # Uruchamianie skryptu do trenowania modelu
    print("Training model...")
    train_process = subprocess.run([venv_python, "train_model.py", model_file])
    if train_process.returncode != 0:
        print("Error occurred during training the model")
        return

    # Uruchamianie skryptu do predykcji
    print("Predicting results...")
    predict_process = subprocess.run([venv_python, "predict_model.py", model_file, output_file])
    if predict_process.returncode != 0:
        print("Error occurred during predicting results")
        return

    print("Both scripts ran successfully!")


if __name__ == "__main__":
    main()
