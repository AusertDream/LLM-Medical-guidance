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

def evaluate(model, tokenizer, modelConfig, prompt, verbose=False):
    """
    获取模型在给定输入下的生成结果。

    参数：
    - query: 用户当前最新prompt
    - chat_history: 聊天记录(不包含RAG内容)
    - generation_config: 模型生成配置。
    - max_len: 最大生成长度。
    - verbose: 是否打印生成结果。

    返回：
    - output: 模型生成的文本。
    """
    model.to()
    model.eval()
    # 设置模型推理时的解码参数
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        num_beams=3,
        top_p=0.3,
        no_repeat_ngram_size=3,
        pad_token_id=0,
    )
    
    # 将提示词转换为模型所需的 token 格式
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 回答
    print("start generating")
    generation_output = model.generate(
        input_ids=inputs["input_ids"],
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=modelConfig["evaluate_max_len"],
        early_stopping=True
    )
    
    # 解码时跳过历史
    response_start = inputs.input_ids.shape[-1]  # 历史部分的 token 长度
    # 解码并打印生成的回复
    full_output = tokenizer.decode(generation_output.sequences[0][response_start:], skip_special_tokens=True)

    return full_output.split("</s>")[0].strip()