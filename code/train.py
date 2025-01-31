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


def startTrain(model, tokenizer, modelConfig):
    wandb.init(project='MedicalGuidance',
               settings=wandb.Settings(start_method='thread', console='off'),
               mode="online",
               name="generatedQA_debug1",
               config={
                   'learning_rate': 3e-4,
                   'architecture': 'Llama3.2-3B',
                   'dataset': 'generatedQA',
                   'batch_size': 4,
                   "epoch": 20,
                   "data_num": 10000
               })
    
    # 创建指定的输出目录
    os.makedirs(modelConfig["output_dir"], exist_ok=True)
    os.makedirs(modelConfig["ckpt_dir"], exist_ok=True)

    # 根据 from_ckpt 标志，从 checkpoint 加载模型权重
    if modelConfig["from_ckpt"]:
        model = PeftModel.from_pretrained(modelConfig["output_dir"])


    # 使用 LoraConfig 配置 LORA 模型
    config = LoraConfig(
        r=modelConfig["LORA_R"],
        lora_alpha=modelConfig["LORA_ALPHA"],
        target_modules=modelConfig["TARGET_MODULES"],
        lora_dropout=modelConfig["LORA_DROPOUT"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # 将 tokenizer 的填充 token 设置为 0
    tokenizer.pad_token_id = 0

    data = load_dataset("json", data_files=modelConfig["dataset_dir"])["train"]

    def generate_training_data(data_point):
        """
        返回：
        - 包含token IDs、labels和attention mask的字典。
        """
    # When users describe their symptoms to you, your task is to accurately determine which hospital department they should visit based on the symptoms provided. If the information given by the user is insufficient to identify the appropriate department, you should ask follow-up questions to gather more detailed symptom information. Your ultimate goal is to help users determine the most suitable medical department for their condition.
    
        prompt = f"""\
    [INST] <<SYS>>
    You are a professional and friendly AI-powered medical triage assistant. 
    <</SYS>>
    Here are some symptoms provided by the patient and the corresponding medical department they should visit in the form of dialog.
    The dialogs are in the list. Per dialog is seperated by a ','.
    learn it.
    {data_point["dialog"]}
    [/INST]"""

        # 计算用户提示词的 token 数量
        len_user_prompt_tokens = (
                len(
                    tokenizer(
                        prompt,
                        truncation=True,
                        max_length=modelConfig["CUTOFF_LEN"] + 1,
                        padding=False,
                    )["input_ids"]
                ) - 1
        )

        # 将完整的输入和输出转换为 tokens
        full_tokens = tokenizer(
            prompt + "</s>",
            truncation=True,
            max_length=modelConfig["CUTOFF_LEN"] + 1,
            padding="max_length",
            )["input_ids"][:-1]
        
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * len(full_tokens),
        }



    data = data.map(generate_training_data, remove_columns=["dialog"])
    
    # train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=modelConfig["LEARNING_RATE"])
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            lossnumber += loss.item()
            wandb.log({"loss_perstep": loss.item()})
            loss.backward()
            optimizer.step()
        
        wandb.log({"loss_perepoch": lossnumber/((len(data)+modelConfig["BATCH_SIZE"] - 1)//modelConfig["BATCH_SIZE"])})
            
    if os.path.exists(modelConfig["output_dir"])==False:
        os.mkdir(modelConfig["output_dir"])
    # 将训练好的模型保存到指定目录
    model.save_pretrained(modelConfig["output_dir"])
    wandb.finish()

    