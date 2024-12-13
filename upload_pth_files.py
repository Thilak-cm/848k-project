import wandb
import os
import re

wandb.init(project="GPT 2 848K Nexus Cluster")

# Create an artifact
artifact = wandb.Artifact("final-models", type="model")

# Find the path for the model folders 
path = os.path.dirname(os.path.abspath(__file__))
if path != '/fs/nexus-scratch/thilakcm/848k-project':
    pattern = r'c848k\d+'
    account = re.findall(pattern, path)[0]
    save_folder = f'/fs/class-projects/fall2024/cmsc848k/{account}'
    os.makedirs(save_folder, exist_ok=True)
else:
    save_folder = '/fs/nexus-scratch/thilakcm'
    os.makedirs(save_folder, exist_ok=True)

# Add files to the artifact from all the folders
for folder in os.listdir(save_folder):
    model_path = f"{save_folder}/{folder}/final_epoch_model.pth"
    if os.path.exists(model_path):
        print(f"{model_path} exists")
        artifact.add_file(local_path=model_path, name=f"{folder}_model.pth")

# Save the artifact
wandb.log_artifact(artifact)

print("Models uploaded successfully!")