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

def evaluate(model, tokenizer, modelConfig, instruction, verbose=False):
    """
    获取模型在给定输入下的生成结果。

    参数：
    - instruction: 描述任务的字符串。
    - generation_config: 模型生成配置。
    - max_len: 最大生成长度。
    - verbose: 是否打印生成结果。

    返回：
    - output: 模型生成的文本。
    """
    # 设置模型推理时的解码参数
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        num_beams=1,
        top_p=0.3,
        no_repeat_ngram_size=3,
        pad_token_id=0,
    )
    # 构建完整的输入提示词
    prompt = f"""\
User: You are a professional and friendly AI-powered medical triage assistant. When users describe their symptoms to you, your task is to accurately determine which hospital department they should visit based on the symptoms provided. If the information given by the user is insufficient to identify the appropriate department, you should ask follow-up questions to gather more detailed symptom information. Your ultimate goal is to help users determine the most suitable medical department for their condition.
{instruction}
Assistant:
"""

    # 将提示词转换为模型所需的 token 格式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    # 使用模型生成回复
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=modelConfig["max_len"],
    )

    # 解码并打印生成的回复
    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)

    return output