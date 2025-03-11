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
data source: github and generated from deepseek
author: Tianyi Jiang
"""

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

    # start train
    train.startTrain(modelConfig)

    print("Training finished.")