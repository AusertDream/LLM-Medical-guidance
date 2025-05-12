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
from transformers import pipeline
from transformers import GenerationConfig

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

model = AutoModelForCausalLM.from_pretrained(modelConfig["evaluate_model"])
# if modelConfig["evaluate_model"] != './model/source':
#     print("load model:", modelConfig["evaluate_model"])
#     model = PeftModel.from_pretrained(model, modelConfig["evaluate_model"]).to(modelConfig["device_map"])
# else:
#     print("load model:", modelConfig["model_name"])
#     model.to(modelConfig["device_map"])
tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])
tokenizer.pad_token = tokenizer.eos_token
generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.3,
        num_beams=3,
        top_p=0.3,
        no_repeat_ngram_size=3,
        pad_token_id=0,
        max_new_tokens=256,
)


app = Flask(__name__)   

# CORS(app, origins=["http://localhost:5173"])
CORS(app, resources={r"/*": {"origins": "*"}})

def get_assistant_format(content):
    return {'role':'assistant', 'content':content}

def get_user_format(content):
    return {'role':'user', 'content':content}

def get_system_format(content):
    return {'role':'system', 'content':content}




def print_gpu_memory():
    print("\n===== GPU 显存监控 =====")
    print(f"当前显存分配量: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"最大显存分配量: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"当前显存保留量: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"最大显存保留量: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
    print("=======================\n")


def refactor_history(chat_history):
    history = []
    instruction = '现在你要扮演一个医院中导诊台的护士，你的职责是根据患者的病情描述，告诉他们应该挂什么科室。如果不足以确认科室，应该向用户进一步提问。所给相关知识仅供参考，具体以患者实际情况为准。要求对话尽量简短。最终必须且只给出一个科室。'
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
            

    

def refactor_prompt(prompt, RAGon=True):
    if RAGon == False:
        return prompt
    else:
        res = """以下为相关参考资料，资料仅供参考，具体以患者实际情况为准。
        【相关参考资料】
        {context}   
        【用户问题】
        {query}""".format(context=get_context(prompt, n_results=3), query=prompt)
        
        return res



@app.route('/generate', methods=['POST'])
def generate():
    json_get = request.json
    if not json_get:
        return jsonify({'error': 'No prompt provided'}), 400

    chat_history = json_get["chatHistory"]
    prompt = json_get["userPrompt"]
    history = refactor_history(chat_history)
    prompt = refactor_prompt(prompt, RAGon=True)
    history.append(get_user_format(prompt))
    print("start generate the answer...")
    model_answer = evaluate.inference_from_transforms(history, generation_config, model, tokenizer, modelConfig["device_map"])
    # print("answer get!:", model_answer)
    return jsonify({'generated_text': model_answer["content"]})

app.run(host='localhost', port=10000)
