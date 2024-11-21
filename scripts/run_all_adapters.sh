#!/bin/bash

# List of adapter and model names
adapters=("houlsby" "pfeiffer" "lora" "ia3" "parallel" "prefixtuning")
models=("codebert" "graphcodebert" "unixcoder" "codeberta" "codet5" "plbart")

# Path to the Python script
script_path="run_scpd.py"

# List of datasets
datasets=(1 2)

# Loop through the datasets, adapters, and models, and execute the Python script
for dataset in "${datasets[@]}"; do
    for adapter in "${adapters[@]}"; do
        for model in "${models[@]}"; do
            echo "Running script for the dataset: $dataset, adapter: $adapter, and model: $model"
            python "$script_path" --dataset_version "$dataset" --model_name "$model" --tune_type peft --epoch 50 --report wandb --adapter_type "$adapter"
            echo "-------------------------------------"
        done
    done
done
