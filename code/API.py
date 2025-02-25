import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from peft import PeftModel
from flask_cors import CORS
import evaluate
import sys

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

model = AutoModelForCausalLM.from_pretrained(modelConfig["model_name"])
if modelConfig["evaluate_model"] != './model/source':
    print("load model:", modelConfig["evaluate_model"])
    model = PeftModel.from_pretrained(model, modelConfig["evaluate_model"]).to(modelConfig["device_map"])
else:
    print("load model:", modelConfig["model_name"])
tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])

app = Flask(__name__)   

# CORS(app, origins=["http://localhost:5173"])
CORS(app, resources={r"/*": {"origins": "*"}})

def add_system_message(prompt, msg):
     return prompt + f"<|start_header_id|>system<|end_header_id|>\n {msg}"

def add_user_message(prompt, msg):
    return prompt + f"<|start_header_id|>user<|end_header_id|>\n User message: \"{msg}\" Tell me which hospital department should I go"

def add_assistant_message(prompt, msg):
     return prompt + f"<|start_header_id|>assistant<|end_header_id|>\n Assistant message: \"{msg}\""


def add_end_token(prompt):
     return prompt + "<|eot_id|>"

def add_start_token(prompt):
     return prompt + "<|begin_of_text|>"

def add_assistant_end_message(prompt):
     return prompt + f"<|start_header_id|>assistant<|end_header_id|>"

def get_instruction(chat_history, user_prompt):
    chat_messages = ""
    sys_message = "You are a helpful assistant in the hospital. You are helping users find the right department to go to."
    chat_messages = add_start_token(chat_messages)
    chat_messages = add_system_message(chat_messages, sys_message)
    chat_messages = add_end_token(chat_messages)
    # chat_messages ="""
    # 你是一个乐于助人的医院智能导诊助手，你的任务是帮助用户找到他们应该去医院挂的科。填充Assistant:后面的内容。
    # """
    for message in chat_history:
        if message["sender"] == "ai":
            chat_messages = add_start_token(chat_messages)
            chat_messages = add_assistant_message(chat_messages, message['text'])
            chat_messages = add_end_token(chat_messages)
        else:
            chat_messages = add_start_token(chat_messages)
            chat_messages = add_user_message(chat_messages, message['text'])
            chat_messages = add_end_token(chat_messages)

    chat_messages = add_start_token(chat_messages)
    chat_messages = add_assistant_end_message(chat_messages)
    return chat_messages
     


@app.route('/generate', methods=['POST'])
def generate():
    json_get = request.json
    if not json_get:
        return jsonify({'error': 'No prompt provided'}), 400

    chat_history = json_get["chatHistory"]
    prompt = json_get["userPrompt"]
    instruction = get_instruction(chat_history, prompt)
    model_answer = evaluate.evaluate(model, tokenizer, modelConfig, instruction)
    return jsonify({'generated_text': model_answer})

app.run(host='localhost', port=10000)
