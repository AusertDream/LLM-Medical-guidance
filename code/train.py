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

# hyperparmas
num_train_data = 1040  # 设置用于训练的数据量，最大值为5000。通常，训练数据越多越好，模型会见到更多样化的诗句，从而提高生成质量，但也会增加训练时间。
# 使用默认参数(1040)：微调大约需要25分钟，完整运行所有单元大约需要50分钟。
# 使用最大值(5000)：微调大约需要100分钟，完整运行所有单元大约需要120分钟。
output_dir = "./output"  # 设置作业结果输出目录。
ckpt_dir = "./exp1"  # 设置 model checkpoint 保存目录（如果想将 model checkpoints 保存到其他目录下，可以修改这里）。
num_epoch = 1  # 设置训练的总 Epoch 数（数值越高，训练时间越长，若使用免费版的 Colab 需要注意时间太长可能会断线，本地运行不需要担心）。
LEARNING_RATE = 3e-4  # 设置学习率
cache_dir = "./cache"  # 设置缓存目录路径
from_ckpt = False  # 是否从 checkpoint 加载模型权重，默认值为否
ckpt_name = None  # 加载特定 checkpoint 时使用的文件名，默认值为无
dataset_dir = "./GenAI-Hw5/Tang_training_data.json"  # 设置数据集目录或文件路径
logging_steps = 20  # 定义训练过程中每隔多少步骤输出一次日志
save_steps = 65  # 定义训练过程中每隔多少步骤保存一次模型
save_total_limit = 3  # 控制最多保留多少个模型 checkpoint
report_to = None  # 设置上报实验指标的目标，默认值为无
MICRO_BATCH_SIZE = 4  # 定义微批次大小
BATCH_SIZE = 16  # 定义一个批次的大小
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 计算每个微批次累积的梯度步骤
CUTOFF_LEN = 256  # 设置文本截断的最大长度
LORA_R = 8  # 设置 LORA（Layer-wise Random Attention）的 R 值
LORA_ALPHA = 16  # 设置 LORA 的 Alpha 值
LORA_DROPOUT = 0.05  # 设置 LORA 的 Dropout 率
VAL_SET_SIZE = 0  # 设置验证集的大小，默认值为无
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]  # 设置目标模块，这些模块的权重将被保存为 checkpoint。
device_map = "auto"  # 设置设备映射，默认值为 "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获取环境变量 "WORLD_SIZE" 的值，若未设置则默认为 1
ddp = world_size != 1  # 根据 world_size 判断是否使用分布式数据处理(DDP)，若 world_size 为 1 则不使用 DDP
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size



cache_dir = "./cache"
model_name = "MediaTek-Research/Breeze-7B-Instruct-v0_1"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 从指定模型名称或路径加载预训练语言模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)

# 创建 tokenizer 并设置结束符号 (eos_token)
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token

# 设置模型推理时的解码参数
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=2,
)


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
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
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
                    max_length=CUTOFF_LEN + 1,
                    padding="max_length",
                )["input_ids"]
            ) - 1
    )

    # 将完整的输入和输出转换为 tokens
    full_tokens = tokenizer(
        prompt + " " + data_point["output"] + "</s>",
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
        )["input_ids"][:-1]

    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * len(full_tokens),
    }

def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    """
    获取模型在给定输入下的生成结果。

    参数：
    - instruction: 描述任务的字符串。
    - generation_config: 模型生成配置。
    - max_len: 最大生成长度。
    - input_text: 输入文本，默认为空字符串。
    - verbose: 是否打印生成结果。

    返回：
    - output: 模型生成的文本。
    """
    # 构建完整的输入提示词
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{instruction}
{input_text}
[/INST]"""

    # 将提示词转换为模型所需的 token 格式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # 使用模型生成回复
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )

    # 解码并打印生成的回复
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split("[/INST]")[1].replace("</s>", "").replace("<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
        if verbose:
            print(output)

    return output


if __name__ == "__main__":
    # 设置TOKENIZERS_PARALLELISM为false，这里简单禁用并行性以避免报错
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 创建指定的输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 根据 from_ckpt 标志，从 checkpoint 加载模型权重
    if from_ckpt:
        model = PeftModel.from_pretrained(model, ckpt_name)

    # 对量化模型进行预处理以进行训练
    model = prepare_model_for_kbit_training(model)

    # 使用 LoraConfig 配置 LORA 模型
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # 将 tokenizer 的填充 token 设置为 0
    tokenizer.pad_token_id = 0

    # 加载并处理训练数据
    with open(dataset_dir, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    with open("tmp_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data_json[:num_train_data], f, indent=2, ensure_ascii=False)

    data = load_dataset('json', data_files="tmp_dataset.json", download_mode="force_redownload")

    # 将训练数据分为训练集和验证集（若 VAL_SET_SIZE 大于 0）
    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_training_data)
        val_data = train_val["test"].shuffle().map(generate_training_data)
    else:
        train_data = data['train'].shuffle().map(generate_training_data)
        val_data = None

    # 使用 Transformers Trainer 进行模型训练
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=50,
            num_train_epochs=num_epoch,
            learning_rate=LEARNING_RATE,
            fp16=True,  # 使用混合精度训练
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=ckpt_dir,
            save_total_limit=save_total_limit,
            ddp_find_unused_parameters=False if ddp else None,  # 是否使用 DDP，控制梯度更新策略
            report_to=report_to,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 禁用模型的缓存功能
    model.config.use_cache = False

    # 若使用 PyTorch 2.0 以上版本且非 Windows 系统，编译模型
    if torch.__version__ >= "2" and sys.platform != 'win32':
        model = torch.compile(model)

    # 开始模型训练
    trainer.train()

    # 将训练好的模型保存到指定目录
    model.save_pretrained(ckpt_dir)

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