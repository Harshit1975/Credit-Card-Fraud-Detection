import subprocess
import os
import sys

def run_command(command, cwd=None):
    print(f"Running: {command}")
    result = subprocess.run(command, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(1)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate Synthetic Data
    run_command("python notebooks/00_generate_data.py", cwd=root_dir)
    
    # Ingest Data
    run_command("python notebooks/01_ingest.py", cwd=root_dir)
    
    # Train Baselines
    run_command("python src/train_baselines.py", cwd=root_dir)
    
    # Tune and Train Model
    run_command("python src/tune_optuna.py", cwd=root_dir)
    
    print("Pipeline completed successfully. Model is ready.")

if __name__ == "__main__":
    main()
