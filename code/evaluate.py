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
from typing import List, Literal, Optional, Tuple, TypedDict


from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."



def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
    """
    Perform text completion for a list of prompts using the language generation model.
    Args:
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
    Returns:
        List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.
    Note:
        This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.
    """
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    generation_tokens, generation_logprobs = self.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        echo=echo,
    )
    if logprobs:
        return [
            {
                "generation": self.tokenizer.decode(t),
                "tokens": [self.tokenizer.decode(x) for x in t],
                "logprobs": logprobs_i,
            }
            for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        ]
    return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
    """
    Generate assistant responses for a list of conversational dialogs using the language generation model.
    Args:
        dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
    Returns:
        List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.
    Raises:
        AssertionError: If the last message in a dialog is not from the user.
        AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.
    Note:
        This method generates assistant responses for the provided conversational dialogs.
        It employs nucleus sampling to introduce controlled randomness in text generation.
        If logprobs is True, token log probabilities are computed for each generated token.
    """
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        prompt_tokens.append(dialog_tokens)
    generation_tokens, generation_logprobs = self.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
    )
    if logprobs:
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t)
                    if not unsafe
                    else UNSAFE_ERROR,
                },
                "tokens": [self.tokenizer.decode(x) for x in t],
                "logprobs": logprobs_i,
            }
            for t, logprobs_i, unsafe in zip(
                generation_tokens, generation_logprobs, unsafe_requests
            )
        ]
    return [
        {
            "generation": {
                "role": "assistant",
                "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
            }
        }
        for t, unsafe in zip(generation_tokens, unsafe_requests)
    ]

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

def inference_from_transforms(messages, generation_config, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=modelConfig["model_name"],
        tokenizer=tokenizer,
        device_map="cuda:3",
    )
    outputs = pipe(
        messages,
        generation_config=generation_config,
    )
    res = outputs[0]["generated_text"][-1]
    return res
