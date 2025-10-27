#!/bin/bash
#SBATCH -A co_rail
#SBATCH -p savio4_gpu
#SBATCH --gres=gpu:A5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 23:00:00
#SBATCH -J psamp_math
#SBATCH --array=0-4%4

set -euo pipefail

# Hugging Face caches to scratch (override by setting SCRATCH_BASE/HF_BASE)
SCRATCH_BASE=${SCRATCH_BASE:-"/global/scratch/users/$USER"}
HF_BASE=${HF_BASE:-"$SCRATCH_BASE/hf"}
mkdir -p "$HF_BASE"/{hub,models,datasets,torch,xdg}
export HF_HOME="$HF_BASE"
export HF_HUB_CACHE="$HF_BASE/hub"
export HF_DATASETS_CACHE="$HF_BASE/datasets"
export TRANSFORMERS_CACHE="$HF_BASE/models"
export TORCH_HOME="$HF_BASE/torch"
export XDG_CACHE_HOME="$HF_BASE/xdg"

# Optional token
# export HF_TOKEN=***YOUR_TOKEN***

# Runtime/env tweaks
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

# Activate env
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate psamp || source activate psamp
else
  source activate psamp || true
fi

# Move to llm_experiments (use Slurm submit directory)
cd "$SLURM_SUBMIT_DIR/llm_experiments"

BATCH_IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEED:-0}
TEMP=${TEMP:-0.25}
MCMC_STEPS=${MCMC_STEPS:-10}
MODEL=${MODEL:-qwen_math}

echo "Running shard BATCH_IDX=${BATCH_IDX} SEED=${SEED} on $(hostname)"
python power_samp_math.py \
  --model "${MODEL}" \
  --temperature "${TEMP}" \
  --mcmc_steps "${MCMC_STEPS}" \
  --device cuda \
  --batch_idx "${BATCH_IDX}" \
  --seed "${SEED}"
