import requests
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys

# 发送http请求

data = {
    "prompt": [{"text": "I have a headache."}]
}

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

model = AutoModelForCausalLM.from_pretrained(modelConfig["output_dir"]).to(modelConfig["device_map"])
tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])

res = evaluate.evaluate(model, tokenizer, modelConfig, "I have a severely stomachache with vomiting and nausea after I ate a bad apple. tell me which hospital department I should visit.")
print(res)
sys.exit(0)

response = requests.post("http://localhost:10000/generate", json=data)

# 检查请求是否成功
if response.status_code == 200:
    print("请求成功！")
    print(response.json())  # 获取响应内容
else:
    print(f"请求失败，状态码：{response.status_code}")