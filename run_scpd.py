import torch
from torch import nn, torch
import random
import numpy as np
import pandas as pd
from datasets import load_dataset,Dataset, DatasetDict
import re, sys, string, argparse, os,csv
import wandb
from pynvml import *
import logging
from pathlib import Path
import evaluate
from transformers import TrainingArguments, EvalPrediction, TrainerCallback, DataCollatorWithPadding,set_seed, TextClassificationPipeline,AutoModel,AutoConfig,AutoTokenizer,Trainer,AutoModelForSequenceClassification,EarlyStoppingCallback,pipeline
from adapters import RobertaAdapterModel,AutoAdapterModel,AdapterTrainer,DoubleSeqBnConfig,SeqBnConfig,LoRAConfig,CompacterConfig,IA3Config,ParBnConfig,SeqBnInvConfig,MAMConfig,UniPELTConfig,DoubleSeqBnInvConfig,PrefixTuningConfig,ConfigUnion,DiReftConfig,LoReftConfig,NoReftConfig
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_choices = {
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
        "unixcoder": "microsoft/unixcoder-base-nine",
        "codeberta":"huggingface/CodeBERTa-small-v1",
        "codet5": "Salesforce/codet5-small",
        "plbart": "uclanlp/plbart-base",
    }

adapter_choices = {
    "houlsby": DoubleSeqBnConfig(),
    "pfeiffer":SeqBnConfig(),
    "lora":LoRAConfig(),
    "ia3":IA3Config(),
    "prefixtuning": PrefixTuningConfig(),
    "parallel":ParBnConfig(),
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Source code plagiairsm detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for reproucability")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dataset_version", type=int, default=1, choices = [1,2])
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learninign Rate")
    parser.add_argument("--model_name", type=str, default='codeberta',choices = list(model_choices.keys()))
    parser.add_argument("--tune_type", type=str, default='fft',help = "Choose the fine tuning type: peft or fft",choices = ["peft","fft"])
    parser.add_argument("--adapter_type", type=str, choices = list(adapter_choices.keys()))
    parser.add_argument("--report", type=str, default='none',help = "Choose whether to report to wandb or not",choices = ["none","wandb"])
    parser.add_argument("--push_adapter", default = False, action="store_true", help = "If added, the adapter would be pushed to HF")
    parser.add_argument("--push_model", default = False, action="store_true", help = "If added, the model would be pushed to HF")
    parser.add_argument("--save_adapter", default = False, action="store_true", help = "If added, the adapter would be pushed to HF")
    parser.add_argument("--save_model", default = False, action="store_true", help = "If added, the model would be pushed to HF")
    parser.add_argument("--adapter_drop", default = True, action="store_true", help = "If added, Robust Adapter Drop will be applied")
    parser.add_argument("--class_weights", default = True, action="store_true", help = "If added, Weighted Loss Function will be used")
    parser.add_argument("--use_callback", default = True, action="store_true", help = "If added, Early Stopping Callback will be used")
    parser.add_argument("--pre_process", default = True, action="store_true", help = "If added, the code will be pre-processed")

    return parser.parse_args()

def set_seeds(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_java_code(code):
    code = re.sub(r"^import\s+.*?;\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code.strip()

def preprocess_dataset(dataset):
    def preprocess_example(example):
        example["code_1"] = preprocess_java_code(example["code_1"])
        example["code_2"] = preprocess_java_code(example["code_2"])
        return example
    return dataset.map(preprocess_example)

def load_dataset_version (args):
    dataset_name = "dataset/version_" + str(args.dataset_version)
    data_files = {"train": dataset_name + "/train.csv","validation": dataset_name + "/validation.csv","test": dataset_name + "/test.csv"}

    train_dataset = load_dataset("csv", data_files = data_files,split = "train")
    valid_dataset = load_dataset("csv", data_files = data_files,split = "validation")
    test_dataset = load_dataset("csv", data_files = data_files,split = "test")

    if args.pre_process:
        train_dataset = preprocess_dataset(train_dataset)
        valid_dataset = preprocess_dataset(valid_dataset)
        test_dataset = preprocess_dataset(test_dataset)

    return train_dataset, valid_dataset, test_dataset

def create_model(args, device,truncation=True, padding="max_length"):
  if args.model_name == "unixcoder":
     tokenizer = AutoTokenizer.from_pretrained(model_choices[args.model_name],
                                            model_max_length =args.max_seq_length)
  else:
      tokenizer = AutoTokenizer.from_pretrained(model_choices[args.model_name])

  config = AutoConfig.from_pretrained(model_choices[args.model_name], device=device, num_labels=2)

  if args.tune_type == 'peft':
    model = AutoAdapterModel.from_pretrained(model_choices[args.model_name], config=config)

  if args.tune_type == 'fft':
    model = AutoModelForSequenceClassification.from_pretrained(
        model_choices[args.model_name],
        config = config)

  return tokenizer, model

def encode_dataset(tokenizer,train_dataset,valid_dataset,test_dataset,max_seq_length):
    
    enc_train = train_dataset.map(lambda e: tokenizer(e['code_1'],e['code_2'],max_length = max_seq_length, truncation=True,padding="max_length"), batched=True)
    enc_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    enc_valid = test_dataset.map(lambda e: tokenizer(e['code_1'],e['code_2'],max_length = max_seq_length,truncation=True,padding="max_length"), batched=True)
    enc_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    enc_test = test_dataset.map(lambda e: tokenizer(e['code_1'],e['code_2'],max_length = max_seq_length,truncation=True,padding="max_length"), batched=True)
    enc_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return enc_train, enc_valid,enc_test

def train_model(args, data_collator,enc_train,enc_test,model_saved_name,model,tokenizer,logger,enc_valid,test_dataset,cw):

    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            loss_func = nn.CrossEntropyLoss(weight=cw)
            loss = loss_func(logits, labels)
            return (loss, outputs) if return_outputs else loss

    class WeightedLossAdapterTrainer(AdapterTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            loss_func = nn.CrossEntropyLoss(weight=cw)
            loss = loss_func(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(p: EvalPrediction):

        df_test = test_dataset.to_pandas()
        labels = p.label_ids
        if args.model_name == "codet5" or args.model_name == "plbart":
            preds = np.argmax(p.predictions[0], axis=1)
        else:
            preds = np.argmax(p.predictions, axis=1)
        clf_metrics = evaluate.combine(["accuracy", "recall", "precision", "f1","buelfhood/fbeta_score"])

        logger.info("Original_labels")
        logger.info(labels)
        logger.info("Predictions")
        logger.info(preds)
        cr = classification_report(y_true=labels, y_pred=preds)
        logger.info("Classification Report")
        logger.info(cr)

        cm = confusion_matrix(y_true=labels, y_pred=preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-plagiarized', 'Plagiarized'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')

        directory_name = "cm_figures"
        if not os.path.exists(directory_name):
           os.makedirs(directory_name)
        plt.savefig(f"{directory_name}/{model_saved_name}_confusion_matrix.png")

        df_test['predicted_label'] = preds
        df_test['is_misclassified'] = df_test['labels'] != df_test['predicted_label']

        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        df_test[df_test['is_misclassified']].to_csv(f"{results_dir}/{model_saved_name}_misclassified_only.csv", index=False)
        met = clf_metrics.compute(predictions=preds, references=labels)
        logger.info("Metrics")
        logger.info(met)

        return met

    #Setting the training arguements
    training_args = TrainingArguments(
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    logging_steps=100,
    #do_eval=True,
    output_dir=model_saved_name + "/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    #For Evaluation
    save_strategy="epoch",
    eval_strategy="epoch",
    #save_steps =100,
    #eval_steps= 100,
    save_total_limit = 1,
    seed=args.seed,
    report_to = args.report,
    load_best_model_at_end=True,
    #metric_for_best_model = "f_beta_score",
    #greater_is_better=Fals
    #include_inputs_for_metrics=True
    )

    if args.tune_type == "peft":
        config =  adapter_choices[args.adapter_type] 
        model.add_adapter(model_saved_name, config = config, overwrite_ok=True)

        model.add_classification_head(
        model_saved_name,
        num_labels=2,   
        #id2label=id2label,
        overwrite_ok=True
        )

        model.train_adapter(model_saved_name)

        if not args.class_weights:
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=enc_train,
                eval_dataset=enc_valid,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                )
        if args.class_weights:
            trainer = WeightedLossAdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=enc_train,
                eval_dataset=enc_valid,
                compute_metrics=compute_metrics,
                data_collator=data_collator)

        class AdapterDropTrainerCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                skip_layers = list(range(np.random.randint(0, 11)))
                kwargs['model'].set_active_adapters(model_saved_name, skip_layers=skip_layers)

            def on_evaluate(self, args, state, control, **kwargs):
                kwargs['model'].set_active_adapters(model_saved_name, skip_layers=None)

        if args.adapter_drop:
            trainer.add_callback(AdapterDropTrainerCallback())
        if args.use_callback:
            trainer.add_callback(EarlyStoppingCallback(5))

    if args.tune_type == "fft":
        if not args.class_weights:
            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=enc_train,
            eval_dataset=enc_valid,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            )

        if args.class_weights:
                trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=enc_train,
                eval_dataset=enc_valid,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                )

    if args.use_callback:
                trainer.add_callback(EarlyStoppingCallback(5))

    is_show_train_gpu = True;
    train_output = trainer.train()
    logger.info(train_output)

    if is_show_train_gpu:
            gpu_benchmark = show_gpu()
            is_show_train_gpu = False

    if args.tune_type == "peft" and args.adapter_type == "prefixtuning":
        model.eject_prefix_tuning(model_saved_name)
    if args.tune_type == "peft" and ( args.adapter_type == "lora" or args.adapter_type == "ia3"):
        model.merge_adapter(model_saved_name)

    pred_output = trainer.evaluate(enc_test,metric_key_prefix="pred")

    logger.info(pred_output)

    if args.push_model:
        trainer.push_to_hub(model_saved_name)
        tokenizer.push_to_hub(model_saved_name)

    if args.push_adapter:
        model.push_adapter_to_hub(model_saved_name,model_saved_name,datasets_tag = "buelfhood/ConPlag_Split")

    if args.save_model:
        trainer.save_model(model_saved_name)
        tokenizer.save_pretrained(model_saved_name)

    if args.save_adapter:
        model.save_adapter(model_saved_name,model_saved_name,with_head=True)

    return train_output, pred_output,gpu_benchmark

def write_to_csv(args,model_saved_name, train_output, pred_output, benchmark,gpu_benchmark, csv_filename):

    csv_filename = csv_filename + ".csv"
    data = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'version': str(args.dataset_version),
        'max_length':args.max_seq_length,
        'tune_type': args.tune_type,
        'model_name': args.model_name,
        'model_saved_name': model_saved_name,
        **train_output.metrics  
        ,**pred_output,
        **benchmark,
        **gpu_benchmark
    }

    directory_name = "csvs"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_exists = os.path.isfile("csvs/results_pred.csv")
    #print(file_exists)
    with open("csvs/" + "results_pred.csv", 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def get_model_parameters(model, required=True):
    if required:
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model_size = sum(p.numel() for p in model.parameters())
    return model_size / 1e+6

def get_model_size(model, required=True):
    if required:
        model_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 * 1024 )
    else:
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024 )
    return model_size #/ 1e+6

def show_gpu():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    total_memory = 0
    total_used = 0

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)

        total_memory += (info.total // 1048576)
        total_used += (info.used // 1048576)

    gpu_benchmark = {
    "name": nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(0)),
    "num:": device_count ,
    "total_memory": total_memory,
    "total_used": total_used,
    "GPU percentage":(total_used/total_memory)*100}

    nvmlShutdown()

    return gpu_benchmark

def get_class_weights(train_dataset):
    traind_ds = train_dataset.to_pandas()
    class_weights = (1 - (traind_ds["labels"].value_counts().sort_index() / len(traind_ds))).values
    class_weights = torch.from_numpy(class_weights).float().to("cuda")
    return class_weights

def main():
    args = parse_arguments()

    directory_name = "logs"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")

    output_filename = args.model_name +"_" + str(args.dataset_version) + "_" + formatted_datetime + ".csv"
    model_saved_name = "ConPlag_" + str(args.dataset_version) + "_" + args.model_name + "_ep" + str(args.epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.learning_rate).replace(".","_") + "_l" + str(args.max_seq_length) + "_s" + str(args.seed)

    if args.tune_type == "peft":
        if not args.adapter_type:
            raise ValueError ("You need to pass an adapter type for PEFT")
        model_saved_name = model_saved_name + "_" + args.adapter_type

    output_filename = model_saved_name + "_" + formatted_datetime 
    
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)

    file = logging.FileHandler("logs/" + output_filename+ ".log")
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s")
    file.setFormatter(formatter)
    logger.addHandler(file)

    logger.info("=" * 20 + " INITIALIZING " + "=" * 20)
    logger.info("model_saved_name")
    logger.info(model_saved_name)
    logger.info("Arguements:")
    logger.info(args)

    if args.report == "wandb":
        with open("wandb_api.key", mode="r", encoding="utf-8") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
        wandb.init(
        project="ConPlag_Experiments_FSE",
        group= args.tune_type + "_version_" + str(args.dataset_version) ,
        name=model_saved_name,
        )

    set_seeds(args.seed)
    train_dataset,valid_dataset,test_dataset = load_dataset_version(args)

    tokenizer, model     = create_model(args,
                                                device=device)

    cw = get_class_weights (train_dataset)

    enc_train, enc_valid, enc_test = encode_dataset(tokenizer,
                                              train_dataset,
                                              valid_dataset,
                                              test_dataset,
                                              args.max_seq_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_output, pred_output, gpu_benchmark = train_model (args,
                           data_collator,
                           enc_train,
                           enc_test,
                           model_saved_name,
                           model,tokenizer,logger,
                           enc_valid,test_dataset,cw
                           )
    logger.info(gpu_benchmark)

    num_param = get_model_parameters(model)
    num_total_param = get_model_parameters(model, required=False)
    param_percentage = (num_param / num_total_param) * 100

    model_size = get_model_size(model, required=False)
    
    if args.save_adapter or args.save_model:
        model_size = round(sum(p.stat().st_size for p in Path(model_saved_name).rglob('*')) / (1024 * 1024),2)

    benchmark = { "actual_trainable_param": num_param
                 , "total_trainable_param" : num_total_param,
                 "trainable_param_percentage" : param_percentage,
                 "model_size_MB":model_size}

    if args.tune_type == "fft":
        benchmark["adapter_type"] = "fft"
    elif args.tune_type == "peft":
        benchmark["adapter_type"] = args.adapter_type

    logger.info("Benchmark")
    logger.info(benchmark)
    write_to_csv(args,model_saved_name,train_output, pred_output,benchmark,gpu_benchmark,output_filename)


if __name__ == "__main__":
    main()
