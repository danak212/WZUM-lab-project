import subprocess
import os

def main():
    # Define the paths for the model and output files
    model_file = "model.pkl"
    output_file = "blaszkiewicz_daniel.json"

    # Define the path to the Python executable in the virtual environment
    venv_python = os.path.join("venv", "Scripts", "python.exe")

    # Train the model by running the train_model.py script
    print("Training model...")
    train_process = subprocess.run([venv_python, "train_model.py", model_file])
    if train_process.returncode != 0:
        print("Error occurred during training the model")
        return

    # Predict the results by running the predict_model.py script
    print("Predicting results...")
    predict_process = subprocess.run([venv_python, "predict_model.py", model_file, output_file])
    if predict_process.returncode != 0:
        print("Error occurred during predicting results")
        return

    print("Both scripts ran successfully!")

if __name__ == "__main__":
    main()
