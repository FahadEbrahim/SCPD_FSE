#!/bin/bash

# List of names
models=("codebert" "graphcodebert" "unixcoder" "codeberta" "codet5" "plbart")

# Path to the Python script
script_path="run_scpd.py"

# List of datasets
datasets=(1 2)

# Loop through the datasets and names, and execute the Python script
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running full fine tuning script for the dataset: $dataset and model: $model"
        python "$script_path" --dataset_version "$dataset" --learning_rate 1e-5 --tune_type fft --epoch 50 --report wandb --model_name "$model"
        echo "-------------------------------------"
    done
done
