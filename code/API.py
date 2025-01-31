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

model = AutoModelForCausalLM.from_pretrained(modelConfig["model_name"])
# model = PeftModel.from_pretrained(base_model, modelConfig["evaluate_model"]).to(modelConfig["device_map"])
tokenizer = AutoTokenizer.from_pretrained(modelConfig["model_name"])


app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

def get_instruction(chat_history, user_prompt):
    chat_messages = ""
    for message in chat_history:
        if message["sender"] == "ai":
            chat_messages += f"Assistant: {message['text']}\n"
        else:
            chat_messages += f"User: {message['text']}\n"
    
    chat_messages += f"User: {user_prompt} Tell me which hospital department should I go.\n"
    chat_messages += "Assistant:"

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
