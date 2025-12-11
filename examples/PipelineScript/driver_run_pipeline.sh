#! /bin/bash

cd /net/noble/vol3/user/ssontha2/classes/CSE583/fish_eeg/results/yash/12.9.2025_pipeline

source /net/gs/vol1/home/ssontha2/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate fishenv

ERR_PATH="/net/noble/vol3/user/ssontha2/classes/CSE583/fish_eeg/results/yash/12.9.2025_pipeline"

qsub -P pinglay_noble -l mem_free=300G \
     -e "$ERR_PATH/run_pipeline.err" \
     -o "$ERR_PATH/run_pipeline.out" \
     run_pipeline.sh