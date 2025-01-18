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







def startTrain(model, tokenizer, modelConfig):


    # 创建指定的输出目录
    os.makedirs(modelConfig["output_dir"], exist_ok=True)
    os.makedirs(modelConfig["ckpt_dir"], exist_ok=True)

    # 根据 from_ckpt 标志，从 checkpoint 加载模型权重
    if modelConfig["from_ckpt"]:
        model = PeftModel.from_pretrained(model, modelConfig["ckpt_name"])


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

    # 加载并处理训练数据
    with open(modelConfig.dataset_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    def generate_training_data(data_point):
        """
        将输入和输出文本转换为模型可读取的 tokens。
        参数：
        - data_point: 包含 "instruction"、"input" 和 "output" 字段的字典。
        返回：
        - 包含token IDs、labels和attention mask的字典。
        """

        prompt = f"""\
    [INST] <<SYS>>
    You are a professional and friendly AI-powered medical triage assistant. When users describe their symptoms to you, your task is to accurately determine which hospital department they should visit based on the symptoms provided. If the information given by the user is insufficient to identify the appropriate department, you should ask follow-up questions to gather more detailed symptom information. Your ultimate goal is to help users determine the most suitable medical department for their condition.
    <</SYS>>
    
    {data_point["instruction"]} 
    {data_point["input"]}
    [/INST]"""

        # 计算用户提示词的 token 数量
        len_user_prompt_tokens = (
                len(
                    tokenizer(
                        prompt,
                        truncation=True,
                        max_length=modelConfig["CUTOFF_LEN"] + 1,
                        padding="max_length",
                    )["input_ids"]
                ) - 1
        )

        # 将完整的输入和输出转换为 tokens
        full_tokens = tokenizer(
            prompt + " " + data_point["output"] + "</s>",
            truncation=True,
            max_length=modelConfig["CUTOFF_LEN"] + 1,
            padding="max_length",
            )["input_ids"][:-1]

        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * len(full_tokens),
        }


    # 使用 Transformers Trainer 进行模型训练
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=modelConfig["batch_size"],
            warmup_steps=50,
            num_train_epochs=modelConfig["num_epoch"],
            learning_rate=modelConfig["LEARNING_RATE"],
            logging_steps=modelConfig["logging_steps"],
            save_strategy="steps",
            save_steps=modelConfig["save_steps"],
            output_dir=modelConfig["ckpt_dir"],
            save_total_limit=modelConfig["save_total_limit"],
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 禁用模型的缓存功能
    model.config.use_cache = False


    # 开始模型训练
    trainer.train()

    # 将训练好的模型保存到指定目录
    model.save_pretrained(modelConfig["output_dir"])

def train():
    wandb.init(project='test',
               settings=wandb.Settings(start_method='thread', console='off'),
               name="sleep",
               config={
                   'learning_rate': 0.01,
                   'architecture': 'CNN',
                   'dataset': 'random',
                   'batch_size': 32,
                   "epoch": 100
               })

    wandb.finish()