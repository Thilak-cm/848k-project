#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos medium
#SBATCH -t 2-00:00:00
#SBATCH --signal=SIGUSR1@90

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_IB_DISABLE=1

cd ~/
source /fs/nexus-scratch/thilakcm/gpt2-venv/bin/activate
echo "venv started"
#srun -u python main.py --test --config cfgs/finetune_modelnet.yaml --exp_name test14_veckm_modelnet40 --ckpts experiments/finetune_modelnet/cfgs/modelnet40_veckm14/ckpt-best.pth
srun -u torchrun --standalone --nproc_per_node=2 /fs/nexus-scratch/thilakcm/848k-project/gpt2_alibi_training.py
echo "ran successfully"