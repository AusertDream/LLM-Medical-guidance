import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    json_get = request.json

    if not json_get:
        return jsonify({'error': 'No prompt provided'}), 400

    inputs = json_get["prompt"]
    generated_text = "我是复读机：" + inputs
    return jsonify({'generated_text': generated_text})

# app.run(host='0.0.0.0', port=10000)

# 发送http请求

data = {
    "prompt": "你好"
}

response = requests.post("http://100.84.89.170:8080/generate", json=data)

# 检查请求是否成功
if response.status_code == 200:
    print("请求成功！")
    print(response.json())  # 获取响应内容
else:
    print(f"请求失败，状态码：{response.status_code}")
