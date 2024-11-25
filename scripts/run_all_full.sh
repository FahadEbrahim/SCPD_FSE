#!/bin/bash

# List of model names
models=("codebert" "graphcodebert" "unixcoder" "codeberta" "codet5" "plbart")

# Path to the Python script
script_path="run_scpd.py"

# List of datasets
datasets=(1 2)

# Loop through the datasets and model names, and execute the Python script
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running full fine tuning script for the dataset: $dataset and model: $model"
        python "$script_path" --dataset_version "$dataset" --learning_rate 1e-5 --tune_type fft --epoch 30 --report wandb --model_name "$model" --use_callback --pre_process --ft_metric f_beta_score
        echo "-------------------------------------"
    done
done
