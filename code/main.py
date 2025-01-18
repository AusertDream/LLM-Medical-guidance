import os
from time import sleep
import wandb
import random
import json
import torch
import torch.nn as nn
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
import train

"""
model: Llama3.2-3B-instruct
data source: github
author: Tianyi Jiang
"""

if __name__ == '__main__':
    with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(modelConfig["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])

    # start train
    train.startTrain(model, tokenizer, modelConfig)