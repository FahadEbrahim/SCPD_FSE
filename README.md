# Classification of Source Code Plagairsm Instances

This repository is the source code and results of the paper:
"The impact of pre-traiend models and adapters for the classification of source code plagairism instances"

## Requirements

The requirements and packages are listed in [`requirements.txt`](requirements.txt) which can be installed using `pip install -r requirements.txt`.

## Dataset

The dataset is available in the `dataset` folder. It has two versions: 1 (raw) and 2 (template-free). The files are in .csv format.
The dataset is called ConPlag. All the credit goes to the authors, the [paper](https://arxiv.org/abs/2303.10763) and the [dataset](https://zenodo.org/records/7332790) are available here. 
In this work, the dataset was split into 70%,15%,15% for training, validation and testing respectively. 

## Running Scripts
The main code is available at `run_scpd.py`. To run the Python script:

A sample run: (available at `scripts/run_exp.sh`)

```shell
python run_scpd.py --epoch 10 --seed 42 --batch_size 32 --max_seq_length 512 --dataset_version 1 --learning_rate 1e-4 --model_name codeberta --tune_type peft --adapter_type lora --report wandb
```

The arguments are:
- Epoch: The number of epochs. The default is 50.
- seed: For randomization. The default is 42.
- batch_size: The batch size. The default is 16.
- max_seq_length: The maximum sequence length for the tokenizer. The default is 512.
- dataset_version: Whether it's 1 or 2. The default is 1.
- learning_rate: The learning rate to be used. The default is 5e-4
- model_name: The model name to be selected. Choices are: [ "codebert", "graphcodebert", "unixcoder", "codeberta", "codet5", "plbart" ]
- tune_type: Whether to fft or peft.
- adapter_type: The adapter to be trained and merged. Choices are: [ "houlsby", "pfeiffer", "lora", "ia3", "parallel", "prefixtuning"]
- report. The default is none. Another choice is the wandb.
Other parameters are availble on the parse_arguments function within the run_scpd.py file. 

There are two other scripts:
1. `scripts/run_all_full.sh`: Runs the Full Fine-Tuning (FFT) on all models for a single dataset version.
2. `scripts/run_all_adapters.sh`: Runs all adapters on all models for a single dataset version.

## The full results
The full results of all experiments are available on `results/results.csv`. 
The logs and results are available on the following [wandb project](https://wandb.ai/fahad-ebrahim/ConPlag_Experiments_FSE?nw=nwuserfahadebrahim). 

### [wandb](https://wandb.ai)
In case of using wandb for experiment tracking setting report to wandb:
The Python file requires the API key to be available in a file called `wandb_api.key` to be available in the root directory.

### Acknowledgement
This repository used codes and intuitions presented in the following repositories: 
1. [Adapters](https://github.com/adapter-hub/adapters) 
2. [PEFT](https://github.com/zwtnju/PEFT/tree/main)
3. [FineTuner](https://github.com/NougatCA/FineTuner)
