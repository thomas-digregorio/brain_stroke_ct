import wandb
import os

def download_best_model(run_name, project_path):
    print(f"Searching for run '{run_name}' in '{project_path}'...")
    api = wandb.Api()
    runs = api.runs(project_path)
    
    target_run = None
    for run in runs:
        if run.name == run_name:
            target_run = run
            break
            
    if target_run:
        print(f"Found run: {target_run.id}")
        print("Downloading best_model.pth...")
        try:
            target_run.file("best_model.pth").download(replace=True, root=".")
            print("Download successful!")
        except Exception as e:
            print(f"Error downloading file: {e}")
            # Fallback: check if it's in a 'files' subdirectory in the artifact
            print("Checking artifacts...")
            artifacts = target_run.logged_artifacts()
            for artifact in artifacts:
                print(f"Artifact: {artifact.name}")
    else:
        print(f"Run '{run_name}' not found.")

if __name__ == "__main__":
    # Hardcoded for this context
    PROJECT = "tad537113-university-of-texas/brain-stroke-ct"
    RUN_NAME = "exalted-sweep-3"
    download_best_model(RUN_NAME, PROJECT)
