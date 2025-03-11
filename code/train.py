import wandb
import os
import sys
import argparse
import json
import warnings
import logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers
from peft import PeftModel
from colorama import Fore, Style

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from datasets import Dataset


def startTrain(modelConfig):
    wandb.init(project='MedicalGuidance',
               settings=wandb.Settings(start_method='thread', console='off'),
               mode="online",
               name="saves/Llama-3.2-3B-Instruct/lora/train_2025-03-05-13-05-45",
               config={
                   'learning_rate': 3e-4,
                   'architecture': 'Llama3.2-3B',
                   'dataset': 'generatedQA',
                   'batch_size': 16,
                   "epoch": 10,
                   "data_num": 1000
               })
    
    # 创建指定的输出目录
    os.makedirs(modelConfig["output_dir"], exist_ok=True)
    os.makedirs(modelConfig["ckpt_dir"], exist_ok=True)



    # 使用 LoraConfig 配置 LORA 模型
    config = LoraConfig(
        r=modelConfig["LORA_R"],
        lora_alpha=modelConfig["LORA_ALPHA"],
        target_modules=modelConfig["TARGET_MODULES"],
        lora_dropout=modelConfig["LORA_DROPOUT"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLM.from_pretrained(modelConfig['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])
    model = get_peft_model(model, config)

    # 将 tokenizer 的填充 token 设置为 0
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data = load_dataset("json", data_files=modelConfig["dataset_dir"])['train']
    # 将数据集划分为训练集和验证集
    if modelConfig["VAL_SET_SIZE"] > 0:
        train_val_split = data.train_test_split(
            test_size=modelConfig["VAL_SET_SIZE"], shuffle=True, seed=42
        )
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]
    else:
        train_data = data
        val_data = None
    

    def generate_training_data(data_point):
        """
        返回：
        - 包含token IDs、labels和attention mask的字典。
        """
        def refactor_dialog(data_point):
            dialog = []
            dialog.append({"role": "system", "content": data_point["system"]})
            for t in data_point["conversations"]:
                if t == data_point["conversations"][-1]:
                    break
                if t['from'] == 'user':
                    dialog.append({"role": "user", "content": t['value']})
                else:
                    dialog.append({"role": "assistant", "content": t['value']})
            return dialog
    
        prompt = tokenizer.apply_chat_template(
            refactor_dialog(data_point),
            tokenize=False,
            add_generation_prompt=True,
        )
        label = tokenizer.apply_chat_template(
            refactor_dialog([data_point["conversations"][-1]]),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token = tokenizer(
                        prompt,
                        truncation=True,
                        max_length=modelConfig["CUTOFF_LEN"] + 1,
                        padding="max_length",
                    )
        label_token = tokenizer(
                        label,
                        truncation=True,
                        max_length=modelConfig["CUTOFF_LEN"] + 1,
                        padding="max_length",
                    )

        return {
            "input_ids": prompt_token["input_ids"],
            "labels": label_token["input_ids"],
            "attention_mask": prompt_token["attention_mask"],
        }


    data = data.map(generate_training_data, remove_columns=["conversations", "system"])
    
    # train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=modelConfig["LEARNING_RATE"])
    criterion  = nn.CrossEntropyLoss()
    model.to(modelConfig["device_map"])
    for epoch in range(modelConfig["num_epoch"]):
        model.train()
        lossnumber = 0
        for i in tqdm(range(0, len(data), modelConfig["BATCH_SIZE"]), desc=f"Epoch {epoch}:"):
            optimizer.zero_grad()
            batch = data[i: i + modelConfig["BATCH_SIZE"]]
            input_ids = torch.tensor(batch["input_ids"]).to(modelConfig["device_map"])
            attention_mask = torch.tensor(batch["attention_mask"]).to(modelConfig["device_map"])
            labels = torch.tensor(batch["labels"]).to(modelConfig["device_map"])
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            lossnumber += loss.item()
            wandb.log({"train/loss": loss.item()})
            loss.backward()
            optimizer.step()
        
        wandb.log({"train/loss_epoch": lossnumber/((len(data)+modelConfig["BATCH_SIZE"] - 1)//modelConfig["BATCH_SIZE"])})
            
    if os.path.exists(modelConfig["output_dir"])==False:
        os.mkdir(modelConfig["output_dir"])
    # 将训练好的模型保存到指定目录
    model.save_pretrained(modelConfig["output_dir"])

    # start validate
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(val_data), modelConfig["BATCH_SIZE"]), desc="Validation:"):
            batch = val_data[i: i + modelConfig["BATCH_SIZE"]]
            input_ids = torch.tensor(batch["input_ids"]).to(modelConfig["device_map"])
            attention_mask = torch.tensor(batch["attention_mask"]).to(modelConfig["device_map"])
            labels = torch.tensor(batch["labels"]).to(modelConfig["device_map"])
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_loss = loss.item()
            wandb.log({"validation/loss": val_loss/modelConfig["BATCH_SIZE"]})

    
    wandb.finish()




