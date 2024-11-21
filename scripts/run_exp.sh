#!/bin/bash

python run_scpd.py --epoch 10 --seed 42 --batch_size 16 --max_seq_length 512 --dataset_version 1 --learning_rate 1e-4 --model_name codeberta --tune_type peft --adapter_type lora --report none
