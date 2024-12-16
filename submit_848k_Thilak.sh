#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --account=class
#SBATCH --partition=class
#SBATCH --qos high
#SBATCH -t 1-00:00:00
#SBATCH --signal=SIGUSR1@90

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_IB_DISABLE=1

cd ~/
source /fs/classhomes/fall2024/cmsc848k/c848k006/gpt2-venv/bin/activate
echo "venv started"
#srun -u python main.py --test --config cfgs/finetune_modelnet.yaml --exp_name test14_veckm_modelnet40 --ckpts experiments/finetune_modelnet/cfgs/modelnet40_veckm14/ckpt-best.pth
srun -u torchrun --standalone --nproc_per_node=4 /fs/classhomes/fall2024/cmsc848k/c848k006/848k-project/gpt2_sinusoidal_regular_att.py
echo "ran successfully"