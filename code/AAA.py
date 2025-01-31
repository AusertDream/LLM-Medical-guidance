import requests
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys

# 发送http请求

data = {
    "chatHistory": [],
    "userPrompt": "I have a headache."
}

print("")
response = requests.post("http://localhost:10000/generate", json=data)

# 检查请求是否成功
if response.status_code == 200:
    print("请求成功！")
    print(response.json())  # 获取响应内容
else:
    print(f"请求失败，状态码：{response.status_code}")