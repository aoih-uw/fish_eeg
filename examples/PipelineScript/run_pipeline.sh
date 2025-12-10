#! /bin/bash

source /net/gs/vol1/home/ssontha2/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate fishenv

cd /net/noble/vol3/user/ssontha2/classes/CSE583/fish_eeg/src/fish_eeg

python pipeline.py \
      --config_path /net/noble/vol3/user/ssontha2/classes/CSE583/fish_eeg/examples/PipelineScript/yash_config.yaml