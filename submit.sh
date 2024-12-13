#!/bin/bash

#SBATCH --nodes=1
##SBATCH --gres=gpu:rtxa5000:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=128gb
#SBATCH --account=class
#SBATCH --partition=class
#SBATCH --qos high
#SBATCH -t 1-00:00:00
#SBATCH --signal=SIGUSR1@90

## SBATCH --nodes=1
## SBATCH --gres=gpu:rtxa6000:1
## SBATCH --ntasks-per-node=1
## SBATCH --mem-per-cpu=16gb
## SBATCH --account=nexus
## SBATCH --partition=tron
## SBATCH --qos medium
## SBATCH -t 2-00:00:00
## SBATCH --signal=SIGUSR1@90

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_IB_DISABLE=1

cd ~/
source /fs/classhomes/fall2024/cmsc848k/c848k017/gpt2-venv/bin/activate
echo "venv started"
# srun -u torchrun --standalone --nproc_per_node=2 /fs/nexus-scratch/thilakcm/848k-project/gpt2_rope_training.py
srun python /fs/classhomes/fall2024/cmsc848k/c848k017/848k-project/upload_pth_files.py
echo "ran successfully"