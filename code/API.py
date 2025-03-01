import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from peft import PeftModel
from flask_cors import CORS
import evaluate
import sys
from RAGInterface import get_context

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

model = AutoModelForCausalLM.from_pretrained(modelConfig["model_name"])
if modelConfig["evaluate_model"] != './model/source':
    print("load model:", modelConfig["evaluate_model"])
    model = PeftModel.from_pretrained(model, modelConfig["evaluate_model"]).to(modelConfig["device_map"])
else:
    print("load model:", modelConfig["model_name"])
    model.to(modelConfig["device_map"])
tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])

app = Flask(__name__)   

# CORS(app, origins=["http://localhost:5173"])
CORS(app, resources={r"/*": {"origins": "*"}})

def get_assistant_format(content):
    return {'role':'assistant', 'content':content}

def get_user_format(content):
    return {'role':'user', 'content':content}

def get_system_format(content):
    return {'role':'system', 'content':content}


def build_prompt(history, new_user_input):
    """
    将对话历史 + 新输入转换为 LLaMA 要求的对话格式
    输入格式示例：
    history = [
        {'role':'system', 'content':'...'},
        {'role':'user', 'content':'...'},
        {'role':'assistant', 'content':'...'},
        ...
    ]
    """
    prompt = ""
    
    # 处理系统指令
    for msg in history:
        if msg['role'] == 'system':
            prompt += f"<<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
            break  # 通常系统指令只保留第一条
    
    # 处理多轮对话
    for msg in history[1:]:  # 跳过系统指令（已单独处理）
        if msg['role'] == 'user':
            prompt += f"[INST] {msg['content']} [/INST]"
        elif msg['role'] == 'assistant':
            prompt += f" {msg['content']} </s><s>"  # 添加对话终止符
    
    # 添加最新用户输入
    prompt += f"[INST] {new_user_input} [/INST]"
    return prompt


def refactor_history(chat_history):
    history = []
    instruction = '现在你要扮演一个医院中导诊台的护士，你的职责是根据患者的病情描述，告诉他们应该挂什么科室。如果病情描述较少，可以继续询问其他症状。要求对话尽量简短。最终必须且只给出一个科室。'
    sys_commd = get_system_format(instruction)
    history.append(sys_commd)
    for chat in chat_history:
        if chat['sender'] == 'ai':
            history.append(get_assistant_format(chat['text']))
        elif chat['sender'] == 'user':
            history.append(get_user_format(chat['text']))
        else:
            print("history has unknown role! Whatsup?")
    
    return history
            

    

def refactor_prompt(prompt, RAGon=False):
    if RAGon == False:
        return
    else:
        res = """【相关知识】
        {context}
        【用户问题】
        {query}""".format(context=get_context(prompt, n_results=4), query=prompt)
        
        return res

     


@app.route('/generate', methods=['POST'])
def generate():
    json_get = request.json
    if not json_get:
        return jsonify({'error': 'No prompt provided'}), 400

    chat_history = json_get["chatHistory"]
    prompt = json_get["userPrompt"]
    chat_history = refactor_history(chat_history)
    prompt = refactor_prompt(prompt, RAGon=True)
    input = build_prompt(chat_history, prompt)
    print("start generate the answer...")
    model_answer = evaluate.evaluate(model, tokenizer, modelConfig, input)
    print("answer get!:", model_answer)    
    return jsonify({'generated_text': model_answer})

app.run(host='localhost', port=10000)
