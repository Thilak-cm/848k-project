import wandb

wandb.init(project="GPT 2 848K Nexus Cluster")

# Create an artifact
artifact = wandb.Artifact("final-models", type="model")

# Add files to the artifact
artifact.add_file("/fs/nexus-scratch/thilakcm/FIRE/final_epoch_model.pth")

# Save the artifact
wandb.log_artifact(artifact)

print("Models uploaded successfully!")