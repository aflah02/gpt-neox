#!/bin/bash
#SBATCH --job-name="neox"
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH -o /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/SLURM_Logs/%x_%j_%A.out
#SBATCH -e /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/SLURM_Logs/%x_%j_%A.err
#SBATCH --time=20:00:00
#SBATCH --exclusive
#SBATCH --mem=0

# Activate Env
source /NS/venvs/work/afkhan/neoxolmo/bin/activate

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Your hostfile creation script from above
echo "Creating hostfile..."
GPUS_PER_NODE=8
mkdir -p /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/SLURM_Configs/Hostfiles
# need to add the current slurm jobid to hostfile name so that we don't add to previous hostfile
hostfile=/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/SLURM_Configs/Hostfiles/hosts_$SLURM_JOBID
# be extra sure we aren't appending to a previous hostfile
rm $hostfile &> /dev/null
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # add a line to the hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done
echo "Hostfile created."

# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/SLURM_Configs/Hostfiles/hosts_$SLURM_JOBID

# Change DIRECTORY to your gpt-neox clone
cd /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox && python ./deepy.py train.py ./configs/llama2/7B.yml ./configs/local_setup_wandb_modified_with_slurm.yml