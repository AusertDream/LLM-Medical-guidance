import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from peft import PeftModel
from flask_cors import CORS
import evaluate

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

model = AutoModelForCausalLM.from_pretrained(modelConfig["model_name"]).to(modelConfig["device_map"])
tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])


app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])



@app.route('/generate', methods=['POST'])
def generate():
    json_get = request.json

    if not json_get:
        return jsonify({'error': 'No prompt provided'}), 400

    messages = json_get["prompt"]
    instruction = messages[len(messages)-1]["text"]
    model_answer = evaluate.evaluate(model, tokenizer, modelConfig, instruction)
    messages = messages.append({"text": f"{model_answer}", "sender": 'ai'})
    return jsonify({'generated_text': model_answer})

app.run(host='localhost', port=10000)
